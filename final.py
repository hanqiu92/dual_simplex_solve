import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
from scikits import umfpack
from scikits.umfpack import UMFPACK_A,UMFPACK_At,UMFPACK_Aat
import time
from enum import Enum,unique

#############################
## 一些与数值误差相关的常数的定义
INF = 1e16
PRIMAL_TOL = 1e-7
PRIMAL_RELA_TOL = 1e-9
DUAL_TOL = 1e-7
CON_TOL = 1e-5
PIVOT_TOL = 1e-5
REMOVE_TOL = 1e-14
ZERO_TOL = 1e-12
#############################

#############################
## 一些用于记录状态的枚举类型
@unique
class VarStatus(Enum):
    AT_UPPER_BOUND = -1 ## 对应U
    AT_LOWER_BOUND = 1 ## 对应L
    OTHER = 0 ## 对应B

@unique
class BoundType(Enum):
    ## 变量的上下界类型
    BOTH_BOUNDED = 3
    UPPER_BOUNDED = 2
    LOWER_BOUNDED = 1
    FREE = 0

@unique
class SolveStatus(Enum):
    '''
    求解状态，每步DS迭代后返回，用于判断后续动作
    '''
    ONGOING = 0 ## 继续迭代
    OPT = 1 ## 最优
    PRIMAL_INFEAS = 2 ## 原始不可行
    DUAL_INFEAS = 3 ## 对偶不可行
    OTHER = -2
    ERR = -1

    ## 以下是一些需要用到的新状态
    REFACTOR = 5 ## LU重分解
    ROLLBACK = 6 ## 回滚
    PHASE1 = 8 ## 需进行第一阶段
    PHASE2 = 9 ## 需进行第二阶段

@unique
class LinearSolver(Enum):
    '''
    用于对线性方程求解/LU分解工具的选用，默认是UMFPACK；
    计算实验表明UMFPACK比SUPERLU快20%以上
    '''
    UMFPACK = 0
    SUPERLU = 1

LINEAR_SOLVER_TYPE = LinearSolver.UMFPACK
#############################

#############################
## 求解过程用到的元素类
class Solution(object):
    def __init__(self,x,lam,s,sign):
        self.x = x ## 原始问题解，实数
        self.lam = lam ## 对偶问题解，实数
        self.s = s ## 对偶问题解，实数
        self.sign = sign ## 基(B,L,U)的标记，值的类型为上面定义的VarStatus

    def copy(self):
        ## 该函数可以复制解，用于需要保留多组解的场景。
        return Solution(x=self.x.copy(),s=self.s.copy(),
                        lam=self.lam.copy(),sign=self.sign.copy())

class Problem(object):
    def __init__(self,A,b,c,l,u,AT=None):
        ## 基本输入：A,b,c,l,u
        ## 对应问题 min c^T x
        ##        s.t. A x = b,
        ##          l <= x <= u.
        ## 为了提高计算效率，A采用稀疏矩阵存储；b,c,l,u则是dense的向量
        self.A,self.b,self.c,self.l,self.u = A,b,c,l,u
        
        ## 保存矩阵A的转置
        if AT is None:
            self.AT = self.A.T
        else:
            self.AT = AT
            
        ## 保存一些后续不会变的中间变量，可以提高后续评估的计算速度
        self.n,self.m = self.A.shape
        self.bounds_gap = self.u - self.l

        self.bool_upper_unbounded = self.u >= INF ## 是否上界无界
        self.bool_lower_unbounded = self.l <= -INF ## 是否下界无界
        self.bool_not_both_bounded = self.bool_upper_unbounded | self.bool_lower_unbounded

        self.bound_type = np.zeros((self.m,),dtype=int) ## 上下界类型
        self.bound_type[self.bool_lower_unbounded & self.bool_upper_unbounded] = BoundType.FREE.value
        self.bound_type[self.bool_lower_unbounded & ~self.bool_upper_unbounded] = BoundType.UPPER_BOUNDED.value
        self.bound_type[~self.bool_lower_unbounded & self.bool_upper_unbounded] = BoundType.LOWER_BOUNDED.value
        self.bound_type[~self.bool_lower_unbounded & ~self.bool_upper_unbounded] = BoundType.BOTH_BOUNDED.value

        self.primal_lower_bound_tol = - (np.abs(self.l) * PRIMAL_RELA_TOL + PRIMAL_TOL)
        self.primal_upper_bound_tol = (np.abs(self.u) * PRIMAL_RELA_TOL + PRIMAL_TOL)
        self.l_margin = self.l + self.primal_lower_bound_tol ## 考虑数值误差的下界
        self.u_margin = self.u + self.primal_upper_bound_tol ## 考虑数值误差的上界

    def copy(self):
        ## 复制函数，in case我们需要保留多组问题
        return Problem(self.A,b=self.b.copy(),c=self.c.copy(),l=self.l.copy(),u=self.u.copy())

    ## **************************
    ## 原始/对偶目标函数的评估
    def eval_primal_obj(self,sol):
        ## z = c^T x
        return np.dot(self.c,sol.x)

    def eval_dual_obj(self,sol):
        ## z = b^T \lambda + u^T s_u + l^T s_l
        return (np.dot(self.b,sol.lam) + \
                np.dot(self.u * (sol.sign == VarStatus.AT_UPPER_BOUND.value),sol.s) + \
                np.dot(self.l * (sol.sign == VarStatus.AT_LOWER_BOUND.value),sol.s))
    
    ## **************************
    ## 原始/对偶线性约束的可行性评估
    def eval_primal_con_infeas(self,sol):
        ## A x - b
        return (self.A._mul_vector(sol.x) - self.b)

    def eval_dual_con_infeas(self,sol):
        ## A^T \lambda + s - c
        return (self.AT._mul_vector(sol.lam) + sol.s - self.c)

    ## **************************
    ## 原始/对偶变量和基的可行性评估

    ## 是否存在原始变量无界，也可以看作是基的对偶可行程度
    def eval_unbnd(self,sol):
        ## x = INF or -INF
        return ( ((self.bool_upper_unbounded) & (sol.sign == VarStatus.AT_UPPER_BOUND.value)) | \
                 ((self.bool_lower_unbounded) & (sol.sign == VarStatus.AT_LOWER_BOUND.value)) )

    ## 解的原始可行程度
    def eval_primal_inf(self,sol):
        ## l <= x <= u
        primal_inf = np.maximum(sol.x - self.u,0) - np.maximum(self.l - sol.x,0)
        bool_unbnd = self.eval_unbnd(sol)
        primal_inf[bool_unbnd] += INF
        return primal_inf

    ## 解的对偶可行程度
    def eval_dual_inf(self,sol):
        ## s_u <= 0, s_l >= 0
        dual_inf = np.maximum(sol.s,0) * (sol.sign == VarStatus.AT_UPPER_BOUND.value) + \
                    np.maximum(-sol.s,0) * (sol.sign == VarStatus.AT_LOWER_BOUND.value)
        bool_unbnd = self.eval_unbnd(sol)
        dual_inf[bool_unbnd] += np.abs(sol.s[bool_unbnd])
        return dual_inf

    ## 基(B,L,U)与解(x,\lambda,s)的一致程度，也可以看作是基的原始可行程度
    def eval_sign(self,sol):
        ## x_L = l_L, x_U = u_U
        bool_sign = np.zeros((self.m,),dtype=bool)
        bool_lower = sol.sign == VarStatus.AT_LOWER_BOUND.value
        bool_upper = sol.sign == VarStatus.AT_UPPER_BOUND.value
        bool_sign[bool_lower] = sol.x[bool_lower] != self.l[bool_lower]
        bool_sign[bool_upper] = sol.x[bool_upper] != self.u[bool_upper]
        return bool_sign

    ## **************************
    ## 计算对偶目标函数对对偶变量s变化的梯度
    def eval_dual_grads(self,sol,lam_grad,s_grad):
        dual_grads = (np.dot(self.b,lam_grad) + \
                    np.dot(self.u * (sol.sign == VarStatus.AT_UPPER_BOUND.value),s_grad) + \
                    np.dot(self.l * (sol.sign == VarStatus.AT_LOWER_BOUND.value),s_grad))
        return dual_grads
        
    ## **************************
    ## 整合上述评估结果，形成字符串输出
    def check_sol_status(self,sol,print_func=None,print_header=''):
        infeas_dict = {'primal':False,'dual':False,'cons':False,'unbnd':False,'sign':False}

        ## 目标函数
        primal_obj,dual_obj = self.eval_primal_obj(sol),self.eval_dual_obj(sol)
        status_str = 'Obj Primal {:.4e} Dual {:.4e}'.format(primal_obj,dual_obj)
        
        ## 原始变量的infeasibility
        primal_inf = self.eval_primal_inf(sol)
        primal_inf_cnt = np.sum((primal_inf > self.primal_upper_bound_tol) | \
                                (primal_inf < self.primal_lower_bound_tol))
        if primal_inf_cnt > 0:
            status_str += '  Primal Inf {:.4e} ({:d})'.format(np.sum(np.abs(primal_inf)),primal_inf_cnt)
            infeas_dict['primal'] = True

        ## 对偶变量的infeasibility
        dual_inf = self.eval_dual_inf(sol)
        dual_inf_cnt = np.sum(np.abs(dual_inf) > DUAL_TOL)
        if dual_inf_cnt > 0:
            status_str += '  Dual Inf {:.4e} ({:d})'.format(np.sum(np.abs(dual_inf)),dual_inf_cnt)
            infeas_dict['dual'] = True

        ## 原始和对偶问题的线性约束的infeasibility
        primal_con_inf,dual_con_inf = self.eval_primal_con_infeas(sol),self.eval_dual_con_infeas(sol)
        con_inf_cnt = np.sum(np.abs(primal_con_inf) > CON_TOL) + np.sum(np.abs(dual_con_inf) > CON_TOL)
        if con_inf_cnt > 0:
            status_str += '  Con Inf {:.4e} ({:d})'.format(np.sum(np.abs(primal_con_inf)) + np.sum(np.abs(dual_con_inf)),con_inf_cnt)
            infeas_dict['cons'] = True

        ## 上下界的consistency
        bool_unbnd = self.eval_unbnd(sol)
        bool_unbnd_cnt = np.sum(bool_unbnd)
        if bool_unbnd_cnt > 0:
            status_str += '  Bnd err {:d}'.format(bool_unbnd_cnt)
            infeas_dict['unbnd'] = True

        ## 解与基的consistency
        bool_sign = self.eval_sign(sol)
        bool_sign_cnt = np.sum(bool_sign)
        if bool_sign_cnt > 0:
            status_str += '  Sign err {:d}'.format(bool_sign_cnt)
            infeas_dict['sign'] = True

        ## 打印输出
        if print_func is not None:
            print_func('{}  {}'.format(print_header,status_str))

        return infeas_dict,status_str

class Basis(object):
    '''
    对$B和N=L or U的元素、以及线性代数相关的计算（相乘/方程求解/LU分解更新）进行管理
    '''
    def __init__(self,A,AT=None):
        self.A = A
        if AT is None:
            self.AT = self.A.T
        else:
            self.AT = AT
        self.n,self.m = self.A.shape

        self.idxB,self.idxN,self.boolN = None,None,None
        self.DSE_weights = None ## DSE权重
        self.invB = None ## LU 分解结果
        self.etas = [] ## PFI的连乘部分
        self.eta_count = 0

    def copy(self):
        basis_new = Basis(self.A,self.AT)
        basis_new.invB = self.invB
        basis_new.eta_count = self.eta_count
        basis_new.idxB = self.idxB.copy()
        basis_new.idxN = self.idxN.copy()
        basis_new.boolN = self.boolN.copy()
        basis_new.DSE_weights = self.DSE_weights.copy()
        basis_new.etas = self.etas.copy()
        return basis_new
    
    def reset_basis_idx(self,idxB):
        self.idxB = idxB.copy()
        self.boolN = np.ones((self.m,),dtype=bool)
        self.boolN[self.idxB] = False
        self.idxN = np.where(self.boolN)[0]
        self.lu_factorize()

    def lu_update(self,eta):
        ## 在PFI中，只需要增加\eta向量和对应的位置i_B即可
        self.etas += [eta]
        self.eta_count += 1

    def lu_factorize(self):
        ## 重新做LU分解
        self.B = self.A[:,self.idxB]
        self.etas = []
        self.eta_count = 0

        if LINEAR_SOLVER_TYPE == LinearSolver.SUPERLU:
            self.invB = splinalg.splu(self.B)
        else:
            self.invB = umfpack.splu(self.B)

    def get_col(self,idx):
        ## 对从矩阵A中获取列idx的封装
        ## 需要A是CSC格式的
        idx_start,idx_end = self.A.indptr[idx],self.A.indptr[idx+1]
        data_,row_ = self.A.data[idx_start:idx_end],self.A.indices[idx_start:idx_end]
        col = np.zeros((self.n,))
        col[row_] = data_
        return col

    def get_elem_vec(self,idx,if_transpose=False):
        ## 对获取单位向量e的封装
        if if_transpose:
            e = np.zeros((self.n,))
        else:
            e = np.zeros((self.m,))
        e[idx] = 1
        return e

    def solve(self,y,if_transpose=False):
        ## 基于PFI的线性方程求解
        if if_transpose:
            ## A_B^T x = y
            y_ = y.copy()
            for eta in self.etas[::-1]:
                y_[eta[0]] += np.dot(eta[1],y_)
            if LINEAR_SOLVER_TYPE == LinearSolver.SUPERLU:
                x = self.invB.solve(y_,trans='T')
            else:
                x = self.invB.umf.solve(UMFPACK_Aat, self.invB._A, y_, autoTranspose=True)
        else:
            ## A_B x = y
            if LINEAR_SOLVER_TYPE == LinearSolver.SUPERLU:
                x = self.invB.solve(y)
            else:
                x = self.invB.umf.solve(UMFPACK_A, self.invB._A, y, autoTranspose=True)
            for eta in self.etas:
                x += x[eta[0]] * eta[1]
        return x

    def dot(self,x,if_transpose=False):
        ## 矩阵与向量的相乘
        if if_transpose:
            ## y = A^T x
            return self.AT._mul_vector(x)
        else:
            ## y = A x
            return self.A._mul_vector(x)

    def get_DSE_weight(self,idx):
        ## 计算idx行的DSE权重|A_B^{-T} e_{idx}|_2^2
        e = self.get_elem_vec(idx,if_transpose=True)
        return np.sum(np.square(self.solve(e,if_transpose=True)))

    def init_DSE_weights(self):
        self.DSE_weights = np.ones((self.n,))

    def reset_DSE_weights(self):
        ## 重新计算DSE权重
        self.DSE_weights = np.array([self.get_DSE_weight(i) for i in range(self.n)])

    def update_DSE_weights(self,idxI,xB_grad0,tau,betaI0):
        ## 通过迭代的方式更新DSE权重
        alpha_j = xB_grad0[idxI]
        betaI = betaI0 / alpha_j / alpha_j
        self.DSE_weights += xB_grad0 * (xB_grad0 * betaI - 2 / alpha_j * tau)
        self.DSE_weights[idxI] = betaI
        self.DSE_weights = np.maximum(self.DSE_weights,1e-4)

    def get_DSE_weights(self):
        return self.DSE_weights

#############################
## 预处理方法

def transform(A,b,sense,c,l,u):
    '''
    将（单侧）不等式约束转换为等式约束
    sense参数指定了约束方向
    '''
    n,m = A.shape
    c = np.concatenate([c,np.zeros((n,))])
    l = np.concatenate([l,np.zeros((n,))])
    u = np.concatenate([u,np.zeros((n,))])
    for colidx in range(n):
        ## 对非等式对应的逻辑变量加上上下界
        if sense[colidx] == 1: ## G
            l[m+colidx] = -INF
        elif sense[colidx] == -1: ## L
            u[m+colidx] = INF
    A = sp.hstack([A,sp.eye(n)],format='csc')
    return A,b,c,l,u

def scaling(A,b,c,l,u):
    '''
    对问题系数进行缩放，以降低条件数和数值难度
    '''

    def scale_row(A,w_row):
        '''
        根据行权重，对稀疏矩阵A的系数进行调整
        假设A是以COO格式存储的
        '''
        w_row[w_row == 0] = 1 ## 如果有权重为0，则重置为1
        A.data /= w_row[A.row] ## 对每个系数，除以相应的行权重
        return A,w_row

    def scale_col(A,w_col):
        '''
        根据列权重，对稀疏矩阵A的系数进行调整
        假设A是以COO格式存储的
        '''
        w_col[w_col == 0] = 1 ## 如果有权重为0，则重置为1
        A.data /= w_col[A.col] ## 对每个系数，除以相应的列权重
        return A,w_col

    def scale(A,w_row,w_col):
        '''
        根据行和列的权重，对稀疏矩阵A的系数进行调整
        假设A是以COO格式存储的
        '''
        A,w_row = scale_row(A,w_row) ## 根据行权重进行调整
        A,w_col = scale_col(A,w_col) ## 根据列权重进行调整
        return A,w_row,w_col

    def l2_scale(A):
        '''
        根据l2范数进行系数调整
        '''
        ## 行和列的权重为l2范数
        A_square = A.multiply(A)
        w_row = np.sqrt(A_square.sum(axis=1).A1)
        w_col = np.sqrt(A_square.sum(axis=0).A1)
        ## 根据行和列的权重，对A的系数进行调整
        A,w_row,w_col = scale(A,np.sqrt(w_row),np.sqrt(w_col))
        return A,w_row,w_col

    def max_scale(A):
        '''
        根据l_{\infty}范数进行系数调整
        '''
        w_row = abs(A).max(axis=1).A[:,0] ## 行权重为每行系数的最大值
        A,w_row = scale_row(A,w_row) ## 根据行权重进行调整
        w_col = abs(A).max(axis=0).A[0,:] ## 列权重为每列系数的最大值
        A,w_col = scale_col(A,w_col) ## 根据列权重进行调整
        return A,w_row,w_col

    def geomean_scale(A):
        '''
        根据几何平均值进行系数调整
        '''
        ## 根据行权重进行调整
        A_abs = abs(A)
        w_row_max = A_abs.max(axis=1).A[:,0] ## 先算出每行系数的最大值
        A_abs.data = 1 / A_abs.data
        w_row_min = A_abs.max(axis=1).A[:,0] ## 再算出每行系数最小值的倒数
        w_row_min[w_row_min == 0] = 1
        w_row = np.sqrt(w_row_max / w_row_min) ## 计算（最大值×最小值）的开方
        A,w_row = scale_row(A,w_row)

        ## 根据列权重进行调整
        A_abs = abs(A)
        w_col_max = A_abs.max(axis=0).A[0,:] ## 先算出每列系数的最大值
        A_abs.data = 1 / A_abs.data
        w_col_min = A_abs.max(axis=0).A[0,:] ## 再算出每列系数最小值的倒数
        w_col_min[w_col_min == 0] = 1
        w_col = np.sqrt(w_col_max / w_col_min)
        A,w_col = scale_col(A,w_col)
        return A,w_row,w_col

    ## scaling的主流程：组合使用多种调整手段
    n,m = A.shape
    A = A.tocoo()
    w_row,w_col = np.ones((n,)),np.ones((m,)) ## 用来保存总体的权重
    A,w_row_tmp,w_col_tmp = l2_scale(A) ## 先根据l2范数进行调整
    w_row,w_col = w_row * w_row_tmp,w_col * w_col_tmp ## 更新总体权重
    A,w_row_tmp,w_col_tmp = max_scale(A) ## 再根据l_{\infty}范数进行调整
    w_row,w_col = w_row * w_row_tmp,w_col * w_col_tmp ## 更新总体权重
    A = A.tocsc()

    ## 在结束对A的更新后，根据总体的行和列权重，对输入向量b,c,l,u的取值进行调整
    ## 注意这里的权重是blog中权重的倒数
    b,c = b / w_row,c / w_col
    l[l > -INF] = l[l > -INF] * w_col[l > -INF]
    u[u < INF] = u[u < INF] * w_col[u < INF]

    return A,b,c,l,u,(w_row,w_col)

def descaling(sol,w_row,w_col):
    '''
    根据预处理时的权重对解进行重缩放处理
    '''
    sol.x /= w_col
    sol.s /= w_col
    sol.lam /= w_row
    return sol

def mat_reduce(A_coo,bool_data_keep,if_inplace=False):
    '''
    对稀疏矩阵A的元素进行删减
    
    输入参数：
    A_coo 以COO格式存储的矩阵
    bool_data_keep 每个元素是否被保留的布尔值
    if_inplace 是否在原矩阵对象中进行操作
    '''
    data,row,col = A_coo.data[bool_data_keep],A_coo.row[bool_data_keep],A_coo.col[bool_data_keep]
    if if_inplace:
        A_coo.data,A_coo.row,A_coo.col = data,row,col
        return A_coo
    else:
        return coo_matrix((data,(row,col)),shape=A_coo.shape)

def rhs_reduce(A_coo,b,x,bool_col_remove):
    '''
    根据要去除的列的集合J，更新右侧项b \gets b - A_J x_J
    
    输入参数：
    A_coo 以COO格式存储的矩阵
    b 右侧项
    x 原始变量
    bool_col_remove 每个列是否需要被去除的布尔值
    '''
    ## 获取矩阵A中需要被去除的元素及相关信息（行i，列j，值A_ij）
    bool_data_remove = bool_col_remove[A_coo.col]
    row,col,data = A_coo.row[bool_data_remove],A_coo.col[bool_data_remove],A_coo.data[bool_data_remove]
    ## 使用bincount函数，将各A_ij x_j的值聚合到行i上
    b -= np.bincount(row,weights=data*x[col],minlength=A_coo.shape[0])
    return b

class Reducer(object):
    def __init__(self,A,b,sense,c,l,u):
        ## 输入初始化
        self.A = A.copy()
        self.A_coo = A.tocoo()
        self.b,self.c = b.copy(),c.copy()
        self.sense,self.l,self.u = sense.copy(),l.copy(),u.copy()
        self.n,self.m = A.shape

        ## 中间变量初始化
        self.x_preprocess = self.l.copy()
        self.x_preprocess[self.l <= -INF] = 0
        self.A_preprocess = []
        self.rows_to_keep = np.ones((self.n,),dtype=bool)
        self.cols_to_keep = np.ones((self.m,),dtype=bool)
        
    def check_empty_row(self):
        '''
        检查空行
        '''
        row_nnz_size = self.A_coo.getnnz(axis=1) ## 获取每个行非零元素的数量
        bool_empty_row = (row_nnz_size == 0) & self.rows_to_keep ## 获取未去除行中的空行
        if np.any(bool_empty_row):
            bool_err = (self.b * self.sense > 0) & bool_empty_row ## 检查可行性
            if np.any(bool_err):
                print('Primal Inf - empty row with inf rhs.')
                exit(1)
                
            self.rows_to_keep = self.rows_to_keep & (~bool_empty_row) ## 保留非空行

    def check_empty_col(self):
        '''
        检查空列
        '''
        col_nnz_size = self.A_coo.getnnz(axis=0) ## 获取每个列非零元素的数量
        bool_empty_col = (col_nnz_size == 0) & self.cols_to_keep ## 获取未去除列中的空列
        if np.any(bool_empty_col):
            bool_c_ge_0,bool_c_le_0 = self.c[bool_empty_col] > 0,self.c[bool_empty_col] < 0
            l_tmp,u_tmp = self.l[bool_empty_col],self.u[bool_empty_col]
            ## 检查对偶可行性
            if np.any( ((bool_c_ge_0) & (l_tmp <= -INF)) | \
                       ((bool_c_le_0) & (u_tmp >= INF))):
                print('Dual Inf - empty col with inf obj.')
                exit(1)
                
            ## 根据目标系数c和上下界确定去除列的x的取值，并保存到x_preprocess中
            x_tmp = l_tmp.copy()
            x_tmp[bool_c_le_0] = u_tmp[bool_c_le_0]
            self.x_preprocess[bool_empty_col] = x_tmp
            self.cols_to_keep = self.cols_to_keep & (~bool_empty_col) ## 保留非空列

    def check_fixed_col(self):
        '''
        检查固定值变量
        '''
        bool_fixed_col = (self.u <= self.l) & self.cols_to_keep ## 用上下界进行判断
        if np.any(bool_fixed_col):
            ## 检查可行性
            if np.any(self.u[bool_fixed_col] < self.l[bool_fixed_col]):
                print('Primal Inf - col with inf bnds.')
                exit(1)
            
            self.x_preprocess[bool_fixed_col] = self.l[bool_fixed_col] ## 将固定值保存到x_preprocess中
            rhs_reduce(self.A_coo,self.b,self.x_preprocess,bool_fixed_col) ## 更新右侧项
            self.cols_to_keep = self.cols_to_keep & (~bool_fixed_col) ## 保留非固定值列
            self.A_coo = mat_reduce(self.A_coo,self.cols_to_keep[self.A_coo.col],if_inplace=True) ## 删减矩阵元素
            
    def check_singleton_row(self):
        '''
        检查单变量约束
        '''
        row_nnz_size = self.A_coo.getnnz(axis=1) ## 获取每个行非零元素的数量
        bool_singleton_row = (row_nnz_size == 1) & self.rows_to_keep ## 获取未去除行中的单元素行
        if np.any(bool_singleton_row):
            ## 获取单元素行的元素及相关信息（行i，列j，值A_ij）
            bool_data = bool_singleton_row[self.A_coo.row] 
            row,col,data = self.A_coo.row[bool_data],self.A_coo.col[bool_data],self.A_coo.data[bool_data]
            sense = self.sense[row] ## 约束方向
            ## 遍历每个行
            for row_,col_,data_,sense_ in zip(row,col,data,sense):
                if ((sense_ >= 0) & (data_ > 0)) | ((sense_ <= 0) & (data_ < 0)):
                    ## 转换为下界约束
                    self.l[col_] = max(self.l[col_],self.b[row_] / data_)
                if ((sense_ <= 0) & (data_ > 0)) | ((sense_ >= 0) & (data_ < 0)):
                    ## 转换为上界约束
                    self.u[col_] = min(self.u[col_],self.b[row_] / data_)
                
            self.rows_to_keep = self.rows_to_keep & (~bool_singleton_row) ## 保留非单变量行
            self.A_coo = mat_reduce(self.A_coo,self.rows_to_keep[self.A_coo.row],if_inplace=True) ## 删减矩阵元素

    def check_obj_col(self):
        '''
        检查最优性隐含变量
        '''
        A_coo = self.A_coo
        ## 计算A_ij * c_j * sense_i的符号，以检查矩阵A中每个元素A_ij和约束方向是否约束变量x_j朝c_j方向前进
        A_tmp = A_coo.copy()
        A_tmp.data = (A_tmp.data * self.sense[A_tmp.row] * self.c[A_tmp.col] >= 0)
        
        bool_no_con_col = np.asarray(A_tmp.sum(axis=0))[0,:] == 0 ## 根据矩阵元素的结果，判断每个列j是否会被约束
        bool_obj_col = bool_no_con_col & self.cols_to_keep ## 获取未去除列中的无约束列
        if np.any(bool_obj_col):
            ## 确定x的值，并将值保存到x_preprocess中
            self.x_preprocess[bool_obj_col] = self.u[bool_obj_col] * (self.c[bool_obj_col] < 0) + \
                                                self.l[bool_obj_col] * (self.c[bool_obj_col] >= 0)
            if np.any(np.abs(self.x_preprocess[bool_obj_col]) >= INF):
                print('Dual Inf - col with inf obj.')
                exit(1)
            rhs_reduce(self.A_coo,self.b,self.x_preprocess,bool_obj_col) ## 更新右侧项
            self.cols_to_keep = self.cols_to_keep & (~bool_obj_col) ## 保留有约束列
            self.A_coo = mat_reduce(self.A_coo,self.cols_to_keep[self.A_coo.col],if_inplace=True) ## 删减矩阵元素

    def check_implied_row(self):
        '''
        检查可行性隐含约束
        '''
        ## 计算每个约束i的隐含上下界b_nega,b_posi
        A_tmp = self.A_coo.tocsr()
        A_posi,A_nega = A_tmp.multiply(A_tmp > 0),A_tmp.multiply(A_tmp < 0)
        b_nega = A_posi.dot(self.l) + A_nega.dot(self.u)
        b_posi = A_posi.dot(self.u) + A_nega.dot(self.l)
        ## 根据b_nega,b_posi和b的相对大小进行判断
        delta_posi,delta_nega = b_posi - self.b,self.b - b_nega
        bool_nega_g,bool_posi_l = delta_nega < -PRIMAL_TOL,delta_posi < -PRIMAL_TOL ## 隐含下(上)界显著大(小)于b
        bool_nega_ge,bool_posi_le = delta_nega <= 0,delta_posi <= 0 ## 隐含下(上)界大(小)于等于b
        bool_nega_eq,bool_posi_eq = delta_nega == 0,delta_posi == 0 ## 隐含下(上)界等于b
        bool_sense_le,bool_sense_ge = self.sense <= 0,self.sense >= 0 ## 约束方向
        ## 检查约束的可行性
        bool_primal_inf = ((bool_sense_le & bool_nega_g) | (bool_sense_ge & bool_posi_l)) & self.rows_to_keep
        if np.any(bool_primal_inf):
            print('Primal Inf - row bnds inf.')
            exit(1)
            
        ## 获取可去除的行以及可固定值的行
        ## 约束 <= 隐含上界 <= b 或者 约束 >= 隐含上界 >= b, 因此sense多余
        bool_redundant_row = ((bool_sense_le & bool_posi_le) | (bool_sense_ge & bool_nega_ge)) & self.rows_to_keep
        bool_fixed_lower_row = bool_sense_le & bool_nega_eq & self.rows_to_keep ## 约束 >= 隐含下界 = b >= 约束
        bool_fixed_upper_row = bool_sense_ge & bool_posi_eq & self.rows_to_keep ## 约束 <= 隐含上界 = b <= 约束
        
        ## 确定所需固定值的变量及固定值
        mat_fixed_lower_row = A_tmp[bool_fixed_lower_row].tocoo()
        mat_fixed_upper_row = A_tmp[bool_fixed_upper_row].tocoo()
        fixed_lower_col = set(mat_fixed_lower_row.col[mat_fixed_lower_row.data >= 0]) | \
                            set(mat_fixed_upper_row.col[mat_fixed_upper_row.data <= 0])
        fixed_upper_col = set(mat_fixed_lower_row.col[mat_fixed_lower_row.data < 0]) | \
                            set(mat_fixed_upper_row.col[mat_fixed_upper_row.data > 0])
        ## 如果有变量同时要固定为上下值，检查可行性
        if len(fixed_lower_col & fixed_upper_col) > 0:
            idx_primal_inf = list(fixed_lower_col & fixed_upper_col)
            if np.sum(self.u[idx_primal_inf] - self.l[idx_primal_inf]) > 0:
                print('Primal Inf - var fixed at both bnds.')

        ## 将固定值存储到x_preprocess中
        fixed_col = list(fixed_lower_col | fixed_upper_col)
        fixed_lower_col,fixed_upper_col = list(fixed_lower_col),list(fixed_upper_col)
        self.x_preprocess[fixed_lower_col] = self.l[fixed_lower_col]
        self.x_preprocess[fixed_upper_col] = self.u[fixed_upper_col]
        ## 更新右侧项
        bool_fixed_col = np.zeros((self.m,),dtype=bool)
        bool_fixed_col[fixed_col] = True
        rhs_reduce(self.A_coo,self.b,self.x_preprocess,bool_fixed_col)
        ## 删除多余的行和列
        self.rows_to_keep = self.rows_to_keep & (~bool_redundant_row)
        self.cols_to_keep[fixed_col] = False
        self.A_coo = mat_reduce(self.A_coo,self.rows_to_keep[self.A_coo.row] & self.cols_to_keep[self.A_coo.col],
                                if_inplace=True) ## 删减矩阵元素

    def check_doubleton_row(self):
        '''
        检查双变量等式约束
        '''
        row_nnz_size = self.A_coo.getnnz(axis=1)  ## 获取每个行非零元素的数量
        bool_doubleton_row = (row_nnz_size == 2) & (self.sense == 0) & self.rows_to_keep ## 获取未去除行中的双元素等式行
        if np.any(bool_doubleton_row):
            ## 遍历每个双变量等式约束
            for row in np.where(bool_doubleton_row)[0]:
                ## 获取该行相关信息
                elem = self.A_coo.row == row
                col = self.A_coo.col[elem]
                if len(col) == 2: ## 确认的确是双变量行
                    ai,bi = self.A_coo.data[elem],self.b[row]
                    ## 根据ai的相对大小进行交换
                    if_first = abs(ai[0]) >= abs(ai[1])
                    col_rm,col_keep = col if if_first else (col[1],col[0])
                    ai_col_rm,ai_col_keep = ai if if_first else (ai[1],ai[0])
                    a_rate,b_rate = ai_col_keep/ai_col_rm,bi/ai_col_rm
                    
                    ## 获取两个列的信息
                    bool_col_rm_data,bool_col_keep_data = self.A_coo.col == col_rm,self.A_coo.col == col_keep
                    col_rm_data,col_rm_row = self.A_coo.data[bool_col_rm_data],self.A_coo.row[bool_col_rm_data]
                    col_keep_data,col_keep_row = self.A_coo.data[bool_col_keep_data],self.A_coo.row[bool_col_keep_data]

                    ## 由于矩阵是稀疏的，通过dict来进行相减以提高效率
                    col_keep_data_by_row = dict()
                    for data_,row_ in zip(col_keep_data,col_keep_row):
                        col_keep_data_by_row[row_] = col_keep_data_by_row.get(row_,0) + data_
                    for data_,row_ in zip(col_rm_data,col_rm_row):
                        col_keep_data_by_row[row_] = col_keep_data_by_row.get(row_,0) - a_rate * data_
                    col_keep_data,col_keep_row = [],[]
                    for row_,data_ in col_keep_data_by_row.items():
                        if abs(data_) > REMOVE_TOL:
                            col_keep_data += [data_]
                            col_keep_row += [row_]
                    col_keep_data = np.array(col_keep_data)
                    col_keep_row = np.array(col_keep_row,dtype=int)
                    col_keep_col = col_keep * np.ones((len(col_keep_row),),dtype=int)

                    ## 根据相减结果更新矩阵
                    bool_data_keep = ~ (bool_col_rm_data | bool_col_keep_data)
                    self.A_coo.data = np.concatenate([self.A_coo.data[bool_data_keep],col_keep_data])
                    self.A_coo.row = np.concatenate([self.A_coo.row[bool_data_keep],col_keep_row])
                    self.A_coo.col = np.concatenate([self.A_coo.col[bool_data_keep],col_keep_col])

                    ## 更新其他输入向量
                    self.b[col_rm_row] -= b_rate * col_rm_data
                    new_bounds = ((bi - ai_col_rm*self.l[col_rm])/ai_col_keep,
                                  (bi - ai_col_rm*self.u[col_rm])/ai_col_keep)
                    new_bounds = new_bounds if new_bounds[1] >= new_bounds[0] else (new_bounds[1],new_bounds[0])
                    self.l[col_keep] = max(self.l[col_keep],new_bounds[0])
                    self.u[col_keep] = min(self.u[col_keep],new_bounds[1])
                    self.c[col_keep] -= a_rate * self.c[col_rm]

                    ## 保留预处理信息，用于后处理
                    self.A_preprocess += [[col_rm,col_keep,ai_col_rm,ai_col_keep,bi]]
                    
                    ## 更新保留行列的信息
                    self.cols_to_keep[col_rm] = False
                    self.rows_to_keep[row] = False
                    
    def drop_row_col(self):
        '''
        根据rows_to_keep和cols_to_keep对各输入矩阵/向量进行化简
        '''
        self.A = self.A_coo.tocsc()
        A = self.A[self.rows_to_keep,:][:,self.cols_to_keep]
        b = self.b[self.rows_to_keep]
        sense = self.sense[self.rows_to_keep]
        c = self.c[self.cols_to_keep]
        l = self.l[self.cols_to_keep]
        u = self.u[self.cols_to_keep]
        return A,b,sense,c,l,u
    
    def reduce(self):
        '''
        简化过程主流程
        '''
        ## 中间变量初始化
        self.x_preprocess = self.l.copy()
        self.x_preprocess[self.l <= -INF] = 0
        self.A_preprocess = []
        self.rows_to_keep = np.ones((self.n,),dtype=bool)
        self.cols_to_keep = np.ones((self.m,),dtype=bool)
        
        ## 循环地组合使用多种简化手段
        tt = time.time()
        curr_size = (np.sum(self.rows_to_keep),np.sum(self.cols_to_keep))
        count_outer,if_diff_outer = 0,True
        while if_diff_outer and count_outer < 10 and (time.time() - tt) < 120:
            count_inner,if_diff_inner = 0,True
            while if_diff_inner and count_inner < 10 and (time.time() - tt) < 120:
                self.check_empty_row() ## 空行
                self.check_empty_col() ## 空列
                self.check_fixed_col() ## 固定值变量
                self.check_obj_col() ## 最优隐含变量
                self.check_singleton_row() ## 单变量约束

                prev_size = curr_size
                curr_size = (np.sum(self.rows_to_keep),np.sum(self.cols_to_keep))
                if_diff_inner = (prev_size != curr_size) ## 检查是否有简化进展
                count_inner += 1

            self.check_implied_row() ## 可行隐含约束
            self.check_doubleton_row() ## 双变量约束
            prev_size = curr_size
            curr_size = (np.sum(self.rows_to_keep),np.sum(self.cols_to_keep))
            if_diff_outer = (prev_size != curr_size) ## 检查是否有简化进展
            count_outer += 1

        ## 完成主要简化逻辑后，再对输入项进行shape上的化简
        A,b,sense,c,l,u = self.drop_row_col()
        return A,b,sense,c,l,u
    
    def restore(self,sol):
        '''
        根据简化过程中的中间结果，对变量进行恢复；这里只考虑对原始变量x的后处理
        '''
        self.x_preprocess[self.cols_to_keep] = sol.x
        for item in self.A_preprocess[::-1]:
            col_rm,col_keep,ai_col_rm,ai_col_keep,bi = item
            self.x_preprocess[col_rm] = (bi - ai_col_keep * self.x_preprocess[col_keep]) / ai_col_rm
        sol.x = self.x_preprocess
        return sol

class Preprocessor(object):
    def __init__(self):
        self.reducer = None
        self.w_row = None
        self.w_col = None
    
    def preprocess(self,A,b,sense,c,l,u):
        '''
        预处理流程
        '''
        ## 首先对问题进行简化
        print('before reduction: ',A.shape,A.nnz)
        tt = time.time()
        self.reducer = Reducer(A,b,sense,c,l,u)
        A,b,sense,c,l,u = self.reducer.reduce()
        print('after reduction: ',A.shape,A.nnz)
        print('reduction process time: {:.2f}'.format(time.time() - tt))

        ## 然后通过系数缩放降低数值难度
        A,b,c,l,u,(w_row,w_col) = scaling(A,b,c,l,u)
        self.w_row,self.w_col = w_row,w_col

        ## 最后进行问题转化
        A,b,c,l,u = transform(A,b,sense,c,l,u)
        return A,b,c,l,u

    def postprocess(self,sol):
        '''
        后处理流程
        '''
        ## 首先去除形式转化后加入的逻辑变量
        n = len(sol.lam)
        m = len(sol.x) - n
        sol.x,sol.s = sol.x[:m],sol.s[:m]
        if self.w_row is not None and self.w_col is not None:
            ## 然后根据权重对各变量取值进行调整
            sol = descaling(sol,self.w_row,self.w_col)
            
        sol.x[np.isnan(sol.x)] = 0
        if self.reducer is not None:
            ## 最后对简化的变量进行恢复
            self.reducer.restore(sol)
        return sol

#############################
## DS求解器

class DualSimplexSolver(object):
    def __init__(self):
        self.global_info = {'count':0,'start_time':time.time(),'phase':1,'fallback_stack':[],'c_raw':None}

    def _get_header(self):
        count = self.global_info.get('count',0)
        phase = self.global_info.get('phase',2)
        header = '{} (P{})'.format(count,phase)
        return header

    def _pricing(self,problem,sol,basis):
        idxB = basis.idxB
        xB = sol.x[idxB]
        primal_inf = np.minimum(xB - problem.l[idxB],0) + np.maximum(xB - problem.u[idxB],0)
        bool_primal_inf = (primal_inf > problem.primal_upper_bound_tol[idxB]) | \
                          (primal_inf < problem.primal_lower_bound_tol[idxB])

        if not np.any(bool_primal_inf):
            ## 原始解可行，因此达到最优
            return SolveStatus.OPT,-1,-1,0

        ## 否则，根据DSE规则选取离开下标idxBI，并保存相应信息
        ## DSE weight已经保存在basis中
        idxI = np.argmax(np.square(primal_inf)/basis.DSE_weights)
        idxBI = idxB[idxI]
        primal_gap = primal_inf[idxI]
        return SolveStatus.ONGOING,idxI,idxBI,primal_gap

    def _ratio_test(self,problem,sol,basis,s_grad,dual_grad):
        '''
        ratio test. 
        
        输入
        problem,sol,basis: 问题、解和基
        s_grad: s的单位变化量
        dual_grad: 对偶目标的单位变化量（即对偶梯度）
        
        输出
        status: 求解状态
        idxJ: 选择的列在N=L\cup U中的位置
        idxNJ: 选择的列在\{1,\cdots,m\}中的位置
        alpha_dual: 对偶步长
        flip_list: 需要翻转类型的列的list
        check_list: 有可能对偶不可行、需要做shift的列的list
        '''
        idxN,boolN = basis.idxN,basis.boolN

        ## 统计可能会约束对偶步长的对偶变量的下标/列
        idxL_bounded = np.where((sol.sign == VarStatus.AT_LOWER_BOUND.value) & (s_grad < -ZERO_TOL))[0]## 处于下界，要求s>=0
        idxU_bounded = np.where((sol.sign == VarStatus.AT_UPPER_BOUND.value) & (s_grad > ZERO_TOL))[0]## 处于上界，要求s<=0
        idxF = np.where((sol.sign == VarStatus.OTHER.value) & boolN)[0]## free变量，要求s==0
        idxF_bounded = idxF[(np.abs(s_grad[idxF]) > ZERO_TOL)]
        elems_bounded = np.concatenate([idxL_bounded,idxU_bounded,idxF_bounded])

        if len(elems_bounded) == 0:
            ## 没有列可以约束对偶步长，因此对偶步长可以无限大，从而对偶目标无界/原始解不可行
            status,idxJ,idxNJ,alpha_dual,flip_list,check_list = SolveStatus.PRIMAL_INFEAS,-1,-1,0,[],[]
            return status,idxJ,idxNJ,alpha_dual,flip_list,check_list

        ## 针对可能约束对偶步长的列，进一步判断其是否可以做bound flip；如果可行，则进行相关计算
        bool_not_both_bounded = problem.bool_not_both_bounded[elems_bounded]
        s_grad_bounded = s_grad[elems_bounded]
        ## 计算bound filp对对偶梯度的影响
        s_grad_abs_bounded = np.abs(s_grad_bounded)
        dual_grad_delta_flipped = problem.bounds_gap[elems_bounded] * s_grad_abs_bounded
        if (np.sum(dual_grad_delta_flipped) <= dual_grad - DUAL_TOL) and (not np.any(bool_not_both_bounded)):
            ## 如果所有约束列都可以做bound flip，而且flip完对偶的梯度仍是正数，则对偶目标无界/原始解不可行
            status,idxJ,idxNJ,alpha_dual,flip_list,check_list = SolveStatus.PRIMAL_INFEAS,-1,-1,0,[],[]
            return status,idxJ,idxNJ,alpha_dual,flip_list,check_list

        ## step 1: 计算每个约束变量对应bound flip的临界对偶步长
        alpha_dual_allowed = - sol.s[elems_bounded] / s_grad_bounded
        alpha_dual_allowed_ub = alpha_dual_allowed + DUAL_TOL / s_grad_abs_bounded

        ## step 2: 通过二分法确定对偶步长上界alpha_dual_ub和对应的列idxs_pivot_ub
        ## 首先，确定alpha_dual_ub搜索的上界alpha_dual_ub_ub和下界alpha_dual_ub_lb
        idxs_pivot_ub = []
        if np.any(bool_not_both_bounded):
            ## 存在不可flip的列，可以大幅降低搜索上界
            alpha_dual_ub_ub = np.min(alpha_dual_allowed_ub[bool_not_both_bounded])
            ## 找到对应最小步长的不可flip列之前的所有列
            idxs_remain = np.where(alpha_dual_allowed_ub < alpha_dual_ub_ub)[0]
            if len(idxs_remain) == 0:
                ## 如果idxs_remain为空，可以直接确定alpha_dual_ub
                alpha_dual_ub = alpha_dual_ub_lb = alpha_dual_ub_ub
                idxs_pivot_ub = list(np.where(alpha_dual_allowed_ub == alpha_dual_ub)[0])
            elif np.sum(dual_grad_delta_flipped[idxs_remain]) <= dual_grad - DUAL_TOL:
                ## 如果idxs_remain全部flip完对偶的梯度仍是正数，可以直接确定alpha_dual_ub
                alpha_dual_ub = alpha_dual_ub_lb = alpha_dual_ub_ub
                idxs_pivot_ub = list(np.where(alpha_dual_allowed_ub == alpha_dual_ub)[0])
            else:
                ## 否则需要进一步搜索
                alpha_dual_ub_lb = np.min(alpha_dual_allowed_ub)
                alpha_dual_ub_ub = np.max(alpha_dual_allowed_ub[idxs_remain])
        else:
            ## 考虑全部列
            alpha_dual_ub_lb = np.min(alpha_dual_allowed_ub)
            alpha_dual_ub_ub = np.max(alpha_dual_allowed_ub)
            idxs_remain = np.arange(len(elems_bounded),dtype=int)

        if len(idxs_pivot_ub) == 0:
            ## 根据区间[alpha_dual_ub_lb,alpha_dual_ub_ub]做二分查找
            dual_grad_tmp = dual_grad - DUAL_TOL
            alpha_dual_ub = (alpha_dual_ub_lb + alpha_dual_ub_ub) / 2
            while len(idxs_remain) > 2 and dual_grad_tmp >= 0 and (alpha_dual_ub_ub - alpha_dual_ub_lb) > 2 * DUAL_TOL / PIVOT_TOL:
                bool_selected = alpha_dual_allowed_ub[idxs_remain] <= alpha_dual_ub
                if not np.any(bool_selected):
                    ## 没有覆盖任何列，直接扩大下界
                    alpha_dual_ub_lb = alpha_dual_ub
                    alpha_dual_ub = (alpha_dual_ub_lb + alpha_dual_ub_ub) / 2
                    continue

                ## 计算dual_grad的变化量
                idxs_selected = idxs_remain[bool_selected]
                delta_dual_grad = np.sum(dual_grad_delta_flipped[idxs_selected])
                if (dual_grad_tmp < delta_dual_grad):
                    ## 超过dual_grad，降低上界
                    idxs_remain = idxs_selected
                    alpha_dual_ub_ub = alpha_dual_ub
                else:
                    ## 不超过dual_grad，扩大下界
                    dual_grad_tmp -= delta_dual_grad
                    idxs_remain = idxs_remain[~bool_selected]
                    alpha_dual_ub_lb = alpha_dual_ub
                alpha_dual_ub = (alpha_dual_ub_lb + alpha_dual_ub_ub) / 2

            if len(idxs_remain) > 1:
                ## 如果二分查找留下多个列未处理，很可能是一个小区间[alpha_dual_ub_lb,alpha_dual_ub_ub]中包含多个列，直接排序搜索
                idxs_remain = idxs_remain[np.argsort(alpha_dual_allowed_ub[idxs_remain])]
                delta_dual_grads = np.cumsum(dual_grad_delta_flipped[idxs_remain])
                idx_first_exceeding = int(np.sum(delta_dual_grads <= dual_grad_tmp)) ## 找到首个超过dual_grad的列
                if idx_first_exceeding < len(idxs_remain):
                    idxs_pivot_ub = [idxs_remain[idx_first_exceeding]]
                else:
                    idxs_pivot_ub = [idxs_remain[-1]]
                alpha_dual_ub = alpha_dual_allowed_ub[idxs_pivot_ub[0]]
            elif len(idxs_remain) == 1:
                ## 如果二分查找后只留下一个列未处理，那么选取该列和对应的对偶步长作为上界
                idxs_pivot_ub = list(idxs_remain)
                alpha_dual_ub = alpha_dual_allowed_ub[idxs_pivot_ub[0]]
            else:
                ## 理论上二分查找后至少有一个列未处理；报错，并当做primal inf退出
                print('error! length of remain idxs = 0.')
                status,idxJ,idxNJ,alpha_dual,flip_list,check_list = SolveStatus.PRIMAL_INFEAS,-1,-1,0,[],[]
                return status,idxJ,idxNJ,alpha_dual,flip_list,check_list

        ## step 3: 根据alpha_dual_ub确定区间，找出所有应考虑的列，并从中选出|s_grad|最大的一个
        alpha_dual_lb = min(alpha_dual_allowed[idxs_pivot_ub])
        idxs_pivot = idxs_pivot_ub
        idxs_selected = np.where((alpha_dual_allowed < alpha_dual_ub) & (alpha_dual_allowed >= alpha_dual_lb))[0]
        if len(idxs_selected) > 0:
            idxs_selected = list(set(list(idxs_selected) + idxs_pivot))
        else:
            idxs_selected = idxs_pivot
        idx_pivot = idxs_selected[np.argmax(s_grad_abs_bounded[idxs_selected])]
        idxNJ = elems_bounded[idx_pivot]
        alpha_dual = alpha_dual_allowed[idx_pivot]

        ## step 4: 进行一些后处理
        if s_grad_abs_bounded[idx_pivot] <= PIVOT_TOL:
            if s_grad_abs_bounded[idx_pivot] <= DUAL_TOL:
                ## pivot size过小，直接进入回滚流程（下面进一步实现）
                status,idxJ,idxNJ,alpha_dual,flip_list,check_list = SolveStatus.ROLLBACK,-1,-1,0,[],[]
                return status,idxJ,idxNJ,alpha_dual,flip_list,check_list

        bool_flip = (alpha_dual_allowed_ub <= alpha_dual)
        flip_list = elems_bounded[bool_flip]
        alpha_dual = max(alpha_dual,ZERO_TOL) ## 扩大对偶步长至某个阈值
        idxJ = np.where(idxN == idxNJ)[0]
        ## 获取需要检查对偶可行性的列
        check_list = elems_bounded[alpha_dual_allowed_ub <= alpha_dual]
        if idxNJ not in check_list:
            check_list = np.array(list(check_list) + [idxNJ])
        return SolveStatus.ONGOING,idxJ,idxNJ,alpha_dual,flip_list,check_list
    
    def _step(self,problem,sol,basis):
        header = self._get_header()

        ## step 1: pricing, 选出离开下标idxBI = idxB[idxI], 并计算相应对偶变量的单位变化量
        status_inner,idxI,idxBI,primal_gap = self._pricing(problem,sol,basis)
        if status_inner != SolveStatus.ONGOING:
            return status_inner,problem,sol,basis
        dual_grad = abs(primal_gap) ## 原始变量的不可行程度正是对偶问题的梯度

        bool_to_lower_bound = sol.x[idxBI] <= problem.l[idxBI]
        direcDualI = 1 if bool_to_lower_bound else -1 ## 原始变量的移动方向
        
        ## 计算对偶变量的单位变化量
        sB_grad0 = basis.get_elem_vec(idxI,if_transpose=True) ## A_B^{-T}e_I
        lam_grad0 = basis.solve(sB_grad0,if_transpose=True) ## A_B^{-T}e_I
        s_grad0 = basis.dot(lam_grad0,if_transpose=True) ## A^TA_B^{-T}e_I
        if direcDualI == -1:
            lam_grad = lam_grad0
            s_grad = -s_grad0
        else:
            lam_grad = -lam_grad0
            s_grad = s_grad0

        ## step 2: ratio test, 选出进入下标idxNJ = idxN[idxJ]
        status_inner,idxJ,idxNJ,alpha_dual,flip_list,check_list = self._ratio_test(problem,sol,basis,s_grad,dual_grad)
        if status_inner != SolveStatus.ONGOING:
            return status_inner,problem,sol,basis

        ## step 3: 更新结果
        aNJ = basis.get_col(idxNJ) ## A_j
        xB_grad0 = basis.solve(aNJ,if_transpose=False) ## A_B^{-1}A_j
        xB_grad = - xB_grad0
        betaI = np.dot(lam_grad0,lam_grad0)
        tau = basis.solve(lam_grad0,if_transpose=False)
        
        ## 校核数值稳定性并进行处理
        if True:
            ## 校核通过\delta s和\delta x_B计算得到的alpha = e_I^T A_B^{-1} a_{NJ}的一致性
            err_pivot = s_grad0[idxNJ] + xB_grad[idxI]
            if abs(err_pivot) > PRIMAL_TOL * (1 + abs(xB_grad[idxI])):
                print('{}  WARN FTRAN/BTRAN pivot consistency err {:.4e}.'.format(header,err_pivot))
                return SolveStatus.REFACTOR,problem,sol,basis
            ## 校核通过\delta s计算得到的e_I^T A_B^T A_B^{-T} e_I = e_I^T e_I = 1的准确性
            err_btran = s_grad0[idxBI] - 1
            if abs(err_btran) > DUAL_TOL:
                print('{}  WARN BTRAN accuracy err {:.4e}.'.format(header,err_btran))
                return SolveStatus.REFACTOR,problem,sol,basis
            ## 校核pivot element的大小
            if abs(xB_grad[idxI]) < PIVOT_TOL / 1e1:
                print('{}  WARN pivot size {:.4e}.'.format(header,xB_grad[idxI]))
                if abs(xB_grad[idxI]) < ZERO_TOL:
                    return SolveStatus.ROLLBACK,problem,sol,basis
        if False:
            ## 校核DSE权重的准确性
            err_dse = betaI - basis.DSE_weights[idxI]
            if abs(err_dse) > PIVOT_TOL * 10:
                print('{}  WARN DSE accuracy err {:.4e}.'.format(header,err_dse))
        
        ## 更新对偶变量
        sol.lam += alpha_dual * lam_grad
        sol.s += alpha_dual * s_grad

        ## 更新原始变量  
        if len(flip_list) > 0:
            ## 对x_N进行翻转
            idx_flip_to_lower = flip_list[sol.sign[flip_list] == VarStatus.AT_UPPER_BOUND.value]
            idx_flip_to_upper = flip_list[sol.sign[flip_list] == VarStatus.AT_LOWER_BOUND.value]
            sol.x[idx_flip_to_lower] = problem.l[idx_flip_to_lower]
            sol.x[idx_flip_to_upper] = problem.u[idx_flip_to_upper]
            sol.sign[idx_flip_to_lower] = VarStatus.AT_LOWER_BOUND.value
            sol.sign[idx_flip_to_upper] = VarStatus.AT_UPPER_BOUND.value
            ## 根据翻转的x_N，更新x_B
            delta_x_flipped = np.zeros((basis.m,))
            delta_x_flipped[idx_flip_to_lower] = -problem.bounds_gap[idx_flip_to_lower]
            delta_x_flipped[idx_flip_to_upper] = problem.bounds_gap[idx_flip_to_upper]
            delta_b_flipped = basis.dot(delta_x_flipped,if_transpose=False)
            delta_xB = - basis.solve(delta_b_flipped,if_transpose=False)
            sol.x[basis.idxB] += delta_xB
            delta_xBI = delta_xB[idxI]
        else:
            delta_xBI = 0

        ## 然后，计算原始步长，并更新x_j和x_B
        alpha_primal = (-primal_gap - delta_xBI) / xB_grad[idxI]
        sol.x[basis.idxB] += alpha_primal * xB_grad
        sol.x[idxBI] = problem.l[idxBI] if bool_to_lower_bound else problem.u[idxBI]
        sol.sign[idxBI] = VarStatus.AT_LOWER_BOUND.value if bool_to_lower_bound else VarStatus.AT_UPPER_BOUND.value
        sol.x[idxNJ] += alpha_primal
        sol.sign[idxNJ] = VarStatus.OTHER.value ## 进入B
        
        ## 检查解的对偶可行性并及时进行shift操作
        if len(check_list) > 0:
            problem,sol = self._shift(problem,sol,check_list=check_list)

        ## 更新基
        basis.idxB[idxI] = idxNJ
        basis.idxN[idxJ] = idxBI
        basis.boolN[idxBI] = True
        basis.boolN[idxNJ] = False
        ## 更新PFI和DSE信息
        eta_vec = -xB_grad0 / xB_grad0[idxI]
        eta_vec[idxI] += 1 / xB_grad0[idxI]
        eta = (idxI,eta_vec)
        basis.lu_update(eta=eta)
        basis.update_DSE_weights(idxI,xB_grad0,tau,betaI)        
        sol.s[basis.idxB] = 0
        
        return SolveStatus.ONGOING,problem,sol,basis

    def _compute_sol_from_basis(self,problem,basis,sign=None):
        '''
        给定一组基，计算对应的解。如果sign没有给出，则这组基是狭义基，不在B中的元素的L/U属性将按照对偶变量s的符号给出
        '''
        idxB,boolN,m = basis.idxB,basis.boolN,basis.m
        
        ## A^T \lambda + s = c, s_B = 0
        lam = basis.solve(problem.c[idxB],if_transpose=True)
        s = problem.c - basis.dot(lam,if_transpose=True)
        s[idxB] = 0
        if sign is None:
            sign = VarStatus.OTHER.value * np.ones((m,),dtype=int)
            sign[boolN & (s < 0)] = VarStatus.AT_UPPER_BOUND.value
            sign[boolN & (s >= 0)] = VarStatus.AT_LOWER_BOUND.value

        ## A_B x_B + A_L x_L + A_U x_U = b, x_L = l_L, x_U = u_U
        x = np.zeros((m,))
        x[sign == VarStatus.AT_LOWER_BOUND.value] = problem.l[sign == VarStatus.AT_LOWER_BOUND.value]
        x[sign == VarStatus.AT_UPPER_BOUND.value] = problem.u[sign == VarStatus.AT_UPPER_BOUND.value]
        x[idxB] = basis.solve(problem.b - basis.dot(x,if_transpose=False),if_transpose=False)

        sol = Solution(x,lam,s,sign)
        return sol

    def _refactorize(self,problem,sol,basis):
        '''
        重新做LU分解并计算解，降低数值误差
        '''
        try:
            basis.lu_factorize()
        except Exception as e:
            print(e)
            return problem,sol,basis
        if sol is not None:
            sol = self._compute_sol_from_basis(problem,basis,sign=sol.sign)
        else:
            sol = self._compute_sol_from_basis(problem,basis)
        return problem,sol,basis
    
    def _perturb(self,problem):
        '''
        针对DS算法的随机扰动方法，参考A. Koberstein论文中的实现
        '''
        ## 先获取原始目标系数
        c = self.global_info['c_raw'].copy()
    
        ## 获取问题属性
        l,u = problem.l,problem.u
        n,m = problem.A.shape
        m_original = m - n

        ## 生成随机扰动的大小
        psi = 1e-5
        perturb_scale = psi * np.abs(c) + 1e2 * DUAL_TOL
        perturb_scale = 0.5 * (1 + np.random.rand(len(c))) * perturb_scale
        ## 根据上下界情况，基于对偶可行性调整扰动方向
        perturb_scale = (1 * (u >= INF) - 1 * (l <= -INF) + 1 * ((u < INF) & (l > -INF))) * perturb_scale

        ## 将扰动值缩放到范围[perturb_min,perturb_max]中
        perturb_min,perturb_max = min(psi,1e-1 * DUAL_TOL),max(psi*np.mean(c),1e2 * DUAL_TOL)
        idx_below = (np.abs(perturb_scale) < perturb_min) & (perturb_scale != 0)
        while np.any(idx_below):
            perturb_scale[idx_below] *= 10
            idx_below = (np.abs(perturb_scale) < perturb_min) & (perturb_scale != 0)
        idx_above = (np.abs(perturb_scale) > perturb_max)
        while np.any(idx_above):
            perturb_scale[idx_above] /= 10
            idx_above = (np.abs(perturb_scale) > perturb_max)

        ## 施加扰动，并保证逻辑变量不受影响
        c_perturb = c + perturb_scale
        c_perturb[m_original:] = 0

        ## 生成扰动后的问题
        problem_perturb = Problem(problem.A,problem.b,c_perturb,problem.l,problem.u)
        
        ## 打印扰动量级
        max_perturb_size = np.max(np.abs(c_perturb - c))
        header = self._get_header()
        print('{}  INFO max perturb size {:.2e}.'.format(header,max_perturb_size))
        return problem_perturb
    
    def _shift(self,problem,sol,check_list=None):
        '''
        对目标系数c进行shift调整，使得对偶解sol.s是可行的；
        参数check_list指定了需要考虑shift的列
        '''
        if check_list is not None:
            s_,sign_ = sol.s[check_list],sol.sign[check_list]
        else:
            s_,sign_ = sol.s,sol.sign
            
        ## 计算解的对偶不可行程度
        bool_lower = sign_ == VarStatus.AT_LOWER_BOUND.value
        bool_upper = sign_ == VarStatus.AT_UPPER_BOUND.value
        dual_inf_ = np.zeros((len(s_),))
        dual_inf_[bool_upper] = np.maximum(s_[bool_upper],0)
        dual_inf_[bool_lower] = np.maximum(-s_[bool_lower],0)
            
        bool_shift = dual_inf_ > DUAL_TOL
        if np.any(bool_shift):
            ## 找到对偶不可行的列，计算应该shift的值
            shift_ = s_[bool_shift]
            # shift_[bool_lower[bool_shift]] += DUAL_TOL
            # shift_[bool_upper[bool_shift]] -= DUAL_TOL
            shift_[bool_lower[bool_shift]] += DUAL_TOL * np.random.rand(np.sum(bool_lower[bool_shift]))
            shift_[bool_upper[bool_shift]] -= DUAL_TOL * np.random.rand(np.sum(bool_upper[bool_shift]))
                
            ## 做shift
            if check_list is not None:
                problem.c[check_list[bool_shift]] -= shift_
                sol.s[check_list[bool_shift]] -= shift_
            else:
                problem.c[bool_shift] -= shift_
                sol.s[bool_shift] -= shift_
                
            ## 如有必要，打印shift量级
            if np.max(np.abs(shift_)) > DUAL_TOL * 10:
                header = self._get_header()
                print('{}  INFO max shift size: {:.2e}.'.format(header,np.max(np.abs(shift_))))
        return problem,sol

    def _save_recent(self,problem,sol,basis,count,phase):
        '''
        将当前解加入回滚堆栈中
        '''
        self.global_info['rollback_stack'] += [(count,phase,basis.copy(),sol.copy())]
        if len(self.global_info['rollback_stack']) > 32:
            self.global_info['rollback_stack'].pop(1)
            
    def _rollback(self,problem,phase):
        '''
        执行回滚操作
        '''
        ## 回滚到堆栈中的最近一个解
        if len(self.global_info['rollback_stack']) > 1:
            last_saved_sol = self.global_info['rollback_stack'].pop()
        else:
            last_saved_sol = self.global_info['rollback_stack'][0]
        last_count,last_phase,basis,sol = last_saved_sol
        header = self._get_header()
        print('{}  INFO rollback to iter {} (P{}).'.format(header,last_count,last_phase))
        basis.lu_factorize() ## 对基进行重分解
        
        ## 重新对问题进行扰动
        problem = self._perturb(problem)
        problem,sol,basis = self._refactorize(problem,sol,basis)
        ## 检查当前解对扰动后的问题的对偶可行性
        dual_inf = problem.eval_dual_inf(sol)
        if np.any(np.abs(dual_inf) > DUAL_TOL * 1e3):
            status = SolveStatus.PHASE1
            return status,problem,sol,basis
        ## 对偶不可行程度低，适当做shift
        problem,sol = self._shift(problem,sol)

        ## 检查回滚前后求解阶段是否一致
        if phase != last_phase:
            ## 求解阶段不一致，则需要跳出当前DS迭代流程
            if last_phase == 1:
                status = SolveStatus.PHASE1
            elif last_phase == 2:
                status = SolveStatus.PHASE2
        else:
            ## 求解阶段一致，可以继续求解
            status = SolveStatus.ONGOING
        return status,problem,sol,basis

    def _loop(self,problem,sol,basis):
        '''
        进行多步迭代，直到求解状态发生变化（非ONGOING）
        加入对迭代步之间的管理
        '''
        count = self.global_info.get('count',0)
        phase = self.global_info.get('phase',2)
        start_time = self.global_info.get('start_time',time.time())
        
        while True:
            header = self._get_header()
            if count % 50000 == 0 and count > 0:
                print('{}  INFO resetting the DSE weights.'.format(header))
                basis.reset_DSE_weights() ## DSE更新

            if basis.eta_count % 20 == 0 and count > 0:
                basis.lu_factorize() ## LU分解
                
            status,problem,sol,basis = self._step(problem,sol,basis) ## 做一步迭代
            count += 1
            self.global_info['count'] = count
            header = self._get_header()

            ## 每隔一定迭代步数监控解的状态
            if ((count % 100 == 0 and count > 0) and (status == SolveStatus.ONGOING)) \
              or (status not in (SolveStatus.ONGOING,SolveStatus.REFACTOR,SolveStatus.ROLLBACK)):
                infeas_dict,status_str = problem.check_sol_status(sol)
                if count % 10000 == 0 and count > 0:
                    print('{}  {}'.format(header,status_str))

                if infeas_dict['unbnd']:
                    ## 基对偶不可行（存在原始变量无界），直接回滚
                    print('{}  WARN Bnd err.'.format(header))
                    status = SolveStatus.ROLLBACK
                elif infeas_dict['sign']:
                    ## 基原始不可行（原始变量与基不一致），尝试根据基重新计算解
                    print('{}  WARN Sign err.'.format(header))
                    status = SolveStatus.REFACTOR
                elif infeas_dict['dual']:
                    ## 解对偶不可行，尝试LU重分解
                    print('{}  WARN Dual inf.'.format(header))
                    status = SolveStatus.REFACTOR
                elif infeas_dict['cons']:
                    ## 线性约束不满足，尝试LU重分解
                    print('{}  WARN Con inf.'.format(header))
                    status = SolveStatus.REFACTOR
                else:
                    if status == SolveStatus.ONGOING:
                        ## 解的状态ok，加入回滚堆栈中
                        if count % 500 == 0:
                            self._save_recent(problem,sol,basis,count,phase)
                    else:
                        ## 进入结束状态，打印并退出
                        print('{}  ({}) {}'.format(header,status.name,status_str))
                        return status,problem,sol,basis
                
            ## 如果状态是LU重分解，进行重分解并分析后续方向
            if status == SolveStatus.REFACTOR:
                problem,sol,basis = self._refactorize(problem,sol,basis)
                infeas_dict,status_str = problem.check_sol_status(sol) ## 检查重分解后解的状态
                if infeas_dict['cons']:
                    ## 约束不满足，说明目前基的条件数过大，有必要进入回滚流程
                    print('{}  WARN Con inf after refactor. cond number of curr basis may be too large.'.format(header))
                    status = SolveStatus.ROLLBACK
                elif infeas_dict['dual'] or infeas_dict['sign']:
                    ## 解或基对偶不可行，重新进入第一阶段
                    print('{}  WARN Dual inf after refactor.'.format(header))
                    status = SolveStatus.PHASE1
                    phase = self.global_info['phase']
                else:
                    ## 解状态正常，继续迭代过程
                    status = SolveStatus.ONGOING

            ## 如果状态是回滚，则进入回滚流程
            if status == SolveStatus.ROLLBACK:
                print('{}  INFO rollback to a recent feasible solution.'.format(header))
                status,problem,sol,basis = self._rollback(problem,phase)

            ## 如果状态非继续迭代，abort
            if status != SolveStatus.ONGOING:
                problem.check_sol_status(sol,print_func=print,print_header=header)
                return status,problem,sol,basis

            ## 限制迭代时长和次数
            if time.time() - start_time > 1.8e4 or count > 1e6:
                print('out of time / iterations.')
                problem.check_sol_status(sol,print_func=print,print_header=header)
                return SolveStatus.OTHER,problem,sol,basis

    def _solve_phase_one(self,problem,sol,basis):
        '''
        基于原始问题构造第一阶段问题（b' \gets 0, l' \gets -I(l = -\infty), u' \gets I(u = \infty)），
        然后调用DS迭代步求解
        '''
        m,n = problem.m,problem.n
        l_,u_ = np.zeros((m,)),np.zeros((m,))
        l_[problem.bool_lower_unbounded] = -100
        u_[problem.bool_upper_unbounded] = 100
        problemPhase1 = Problem(problem.A,np.zeros((n,)),problem.c,l_,u_)
        problemPhase1,solPhase1,basis = self._refactorize(problemPhase1,sol=None,basis=basis)
        statusPhase1,problemPhase1,solPhase1,basis = self._loop(problemPhase1,solPhase1,basis)
        return statusPhase1,problemPhase1,solPhase1,basis
    
    def _solve_phase_two(self,problem,sol,basis):
        problem,sol,basis = self._refactorize(problem,sol,basis)
        status,problem,sol,basis = self._loop(problem,sol,basis)
        return status,problem,sol,basis

    def _solve(self,problem,sol,basis):        
        ## 检查初始基/解的状态
        infeas_dict,status_str = problem.check_sol_status(sol)
        if infeas_dict['unbnd']:
            ## 基对偶不可行，进入第一阶段
            status = SolveStatus.PHASE1
        elif infeas_dict['cons']:
            ## 解线性约束不可行，意味着初始条件数就很大；目前的求解流程无法处理，直接退出
            ## 注：这需要放在基对偶可行判断的后面，因为如果有原始变量无界，由于数值原因，解的线性约束很容易不可行
            print('Cond num of the initial basis is too large. Abort.')
            status = SolveStatus.ERR
        else:
            ## 可直接进入第二阶段
            status = SolveStatus.PHASE2

        ## 进入循环，在两阶段中跳转
        iter_count = 0
        while ((status in (SolveStatus.PHASE1,SolveStatus.PHASE2)) and (iter_count < 100)):
            iter_count += 1
            if status == SolveStatus.PHASE2:
                ## 进入第二阶段，寻找最优基/解
                self.global_info['phase'] = 2
                status,problem,sol,basis = self._solve_phase_two(problem,sol,basis)
            elif status == SolveStatus.PHASE1:
                ## 进入第一阶段，寻找可行基/解
                self.global_info['phase'] = 1
                statusPhase1,problemPhase1,solPhase1,basis = self._solve_phase_one(problem,sol,basis)
                ## 用第一阶段所得的基更新原始问题的解
                problem.c = problemPhase1.c
                sol.sign = solPhase1.sign.copy()

                ## 对第一阶段求解结果进行分析
                if statusPhase1 == SolveStatus.DUAL_INFEAS:
                    ## 如果返回的状态是对偶不可行，则求解流程有问题，直接退出
                    status = SolveStatus.ERR
                    # status = SolveStatus.PHASE1
                elif statusPhase1 in (SolveStatus.OPT,SolveStatus.PRIMAL_INFEAS):
                    ## 如果返回的状态是最优或原始不可行，再次检查当前基是否满足对偶可行性（不存在原始变量无界）
                    bool_unbnd = problem.eval_unbnd(sol)
                    if not np.any(bool_unbnd):
                        ## 基对偶可行，进入第二阶段
                        status = SolveStatus.PHASE2
                    else:
                        ## 基对偶不可行，说明第一阶段求解结果有瑕疵；尝试直接调整基来恢复对偶可行性，但后续需要检查解的对偶可行性
                        bool_resolve_phase1 = False
                        idxs = np.where(bool_unbnd)[0]
                        for idx in idxs:
                            if problem.bound_type[idx] == BoundType.FREE.value:
                                ## free变量无法通过调整基来恢复对偶可行性；
                                ## 直接重新进入第一阶段
                                bool_resolve_phase1 = True
                                break
                            elif problem.bound_type[idx] == BoundType.UPPER_BOUNDED.value:
                                sol.sign[idx] = VarStatus.AT_UPPER_BOUND.value
                            elif problem.bound_type[idx] == BoundType.LOWER_BOUNDED.value:
                                sol.sign[idx] = VarStatus.AT_LOWER_BOUND.value
                        if bool_resolve_phase1:
                            ## 重新进入第一阶段；尝试进行随机扰动
                            problem = self._perturb(problem)
                            status = SolveStatus.PHASE1
                        else:
                            ## 根据调整后的基计算解，并检查其状态
                            sol = self._compute_sol_from_basis(problem,basis,sign=sol.sign)
                            infeas_dict,status_str = problem.check_sol_status(sol)
                            if infeas_dict['sign'] or infeas_dict['unbnd']:
                                ## 基对偶或原始不可行；
                                ## 调整基后一般不会进入这里；一旦进入了，直接退出求解流程
                                print('ERR Basis Primal/Dual Inf. Abort.')
                                status = SolveStatus.ERR
                            elif infeas_dict['dual']:
                                ## 重新进入第一阶段；尝试进行随机扰动
                                problem = self._perturb(problem)
                                status = SolveStatus.PHASE1
                            elif infeas_dict['primal']:
                                status = SolveStatus.PHASE2
                            else:
                                status = SolveStatus.OPT
                else:
                    ## 返回的状态不是标准结束状态（OPT、PRIMAL_INFEAS、DUAL_INFEAS），则直接复制状态并进行随机扰动
                    problem = self._perturb(problem)
                    status = statusPhase1
        return status,problem,sol,basis
            
    def solve(self,A_raw,b_raw,sense_raw,c_raw,l_raw,u_raw):
        '''
        主求解入口
        '''
        self.global_info = {'count':0,'start_time':time.time(),'phase':1,'fallback_stack':[],'c_raw':None}
        
        ## 预处理
        self.preprocessor = Preprocessor()
        A,b,sense,c,l,u = A_raw.copy(),b_raw.copy(),sense_raw.copy(),c_raw.copy(),l_raw.copy(),u_raw.copy()
        A,b,c,l,u = self.preprocessor.preprocess(A,b,sense,c,l,u)

        ## 初始化
        problem = Problem(A,b,c,l,u)
        self.global_info['c_raw'] = problem.c.copy()
        problem = self._perturb(problem)
        n,m = A.shape
        idxB = np.arange(m-n,m,1,dtype=int)
        basis = Basis(A)
        basis.init_DSE_weights()
        basis.reset_basis_idx(idxB)
        problem,sol,basis = self._refactorize(problem,sol=None,basis=basis)
        self.global_info['rollback_stack'] = [(0,1,basis.copy(),sol.copy())]

        ## 开始DS迭代流程
        status,problem,sol,basis = self._solve(problem,sol,basis)
        
        ## 求解完成后做后处理
        sol = self.preprocessor.postprocess(sol)

        return status,sol,basis