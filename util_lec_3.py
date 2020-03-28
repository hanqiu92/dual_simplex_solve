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


class DualSimplexSolver(object):
    def __init__(self):
        ## 保存迭代过程中的信息，用于进行控制流处理
        self.global_info = {'count':0,'start_time':time.time()}

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
        idxN,boolN = basis.idxN,basis.boolN 
        
        ## 统计可能会约束对偶步长的对偶变量的下标
        idxL_bounded = np.where((sol.sign == VarStatus.AT_LOWER_BOUND.value) & (s_grad < 0))[0] ## 处于下界，要求s>=0
        idxU_bounded = np.where((sol.sign == VarStatus.AT_UPPER_BOUND.value) & (s_grad > 0))[0] ## 处于上界，要求s<=0
        idxF = np.where((sol.sign == VarStatus.OTHER.value) & boolN)[0] ## free变量，要求s==0
        idxF_bounded = idxF[(np.abs(s_grad[idxF]) > 0)]
        elems_bounded = np.concatenate([idxL_bounded,idxU_bounded,idxF_bounded])

        if len(elems_bounded) == 0:
            ## 没有变量可以约束对偶步长，因此对偶步长可以无限大，从而对偶目标无界/原始解不可行
            status,idxJ,idxNJ,alpha_dual,flip_list = SolveStatus.PRIMAL_INFEAS,-1,-1,0,[]
            return status,idxJ,idxNJ,alpha_dual,flip_list

        ## 针对可能约束对偶步长的变量，进一步判断其是否可以做bound flip；如果可行，则进行相关计算
        bool_not_both_bounded = problem.bool_not_both_bounded[elems_bounded]
        s_grad_bounded = s_grad[elems_bounded]
        ## 计算bound filp对对偶梯度的影响
        s_grad_abs_bounded = np.abs(s_grad_bounded)
        dual_grad_delta_flipped = problem.bounds_gap[elems_bounded] * s_grad_abs_bounded
        if (np.sum(dual_grad_delta_flipped) <= dual_grad) and (not np.any(bool_not_both_bounded)):
            ## 如果所有约束变量都可以做bound flip，而且flip完对偶的梯度仍是正数，则对偶目标无界/原始解不可行
            status,idxJ,idxNJ,delta_dual,flip_list = SolveStatus.PRIMAL_INFEAS,-1,-1,0,[]
            return status,idxJ,idxNJ,delta_dual,flip_list

        ## 计算每个约束变量对应bound flip的临界对偶步长
        alpha_dual_allowed = - sol.s[elems_bounded] / s_grad_bounded
        
        ## 通过不可flip的变量进一步筛选
        if np.any(bool_not_both_bounded):
            alpha_dual_ub = np.min(alpha_dual_allowed[bool_not_both_bounded])
            ## 找到对应最小步长的不可flip变量之前的所有变量
            idxs_remain = np.where(alpha_dual_allowed <= alpha_dual_ub)[0] 
        else:
            ## 考虑全部变量
            idxs_remain = np.arange(len(elems_bounded),dtype=int)

        ## 通过alpha_dual_allowed对各变量进行排序
        idxs_remain = idxs_remain[np.argsort(alpha_dual_allowed[idxs_remain])]
        ## 做线搜索，寻找临界的变量
        dual_grad_remain = dual_grad
        for idx_pivot in idxs_remain:
            dual_grad_remain -= dual_grad_delta_flipped[idx_pivot]
            if dual_grad_remain < 0:
                break
        
        ## 整理结果
        idxNJ = elems_bounded[idx_pivot]
        alpha_dual = alpha_dual_allowed[idx_pivot]
        bool_flip = (alpha_dual_allowed < alpha_dual)
        flip_list = elems_bounded[bool_flip]
        idxJ = np.where(idxN == idxNJ)[0]
        return SolveStatus.ONGOING,idxJ,idxNJ,alpha_dual,flip_list
    
    def _step(self,problem,sol,basis):
        count = self.global_info.get('count',0)
        header = '{} '.format(count)

        ## step 1: pricing, 选出离开下标idxBI = idxB[idxI], 并计算相应对偶变量的单位变化量
        status_inner,idxI,idxBI,primal_gap = self._pricing(problem,sol,basis)
        if status_inner == SolveStatus.OPT:
            return SolveStatus.OPT,problem,sol,basis
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
        status_inner,idxJ,idxNJ,alpha_dual,flip_list = self._ratio_test(problem,sol,basis,s_grad,dual_grad)
        if status_inner == SolveStatus.PRIMAL_INFEAS:
            return SolveStatus.PRIMAL_INFEAS,problem,sol,basis

        ## step 3: 更新结果
        
        aNJ = basis.get_col(idxNJ) ## A_j
        xB_grad0 = basis.solve(aNJ,if_transpose=False) ## A_B^{-1}A_j
        xB_grad = - xB_grad0
        betaI = np.dot(lam_grad0,lam_grad0)
        tau = basis.solve(lam_grad0,if_transpose=False)
        
        ## 校核数值稳定性，在这一个notebook中只做评估而不进行处理
        if True:
            ## 校核通过\delta s和\delta x_B计算得到的alpha = e_I^T A_B^{-1} a_{NJ}的一致性
            err_pivot = s_grad0[idxNJ] + xB_grad[idxI]
            if abs(err_pivot) > PRIMAL_TOL * (1 + abs(xB_grad[idxI])):
                print('{}  WARN err FTRAN/BTRAN pivot consistency {:.4e}.'.format(header,err_pivot))
            ## 校核DSE权重的准确性
            err_dse = betaI - basis.DSE_weights[idxI]
            if abs(err_dse) > PIVOT_TOL * 10:
                print('{}  WARN err DSE accuracy {:.4e}.'.format(header,err_dse))
            ## 校核pivot element的大小
            if abs(xB_grad[idxI]) < PIVOT_TOL:
                print('{}  WARN err pivot size {:.4e}.'.format(header,xB_grad[idxI]))
        
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
    
    def _loop(self,problem,sol,basis):
        '''
        进行多步迭代，直到求解状态发生变化（非ONGOING）
        加入对迭代步之间的管理
        '''
        count = self.global_info.get('count',0)
        start_time = self.global_info.get('start_time',time.time())
        while True:
            if count % 5000 == 0 and count > 0:
                print('resetting the DSE weights!')
                basis.reset_DSE_weights() ## DSE更新
            if basis.eta_count % 20 == 0 and count > 0:
                basis.lu_factorize() ## LU分解
                
            status,problem,sol,basis = self._step(problem,sol,basis) ## 做一步迭代
            count += 1
            self.global_info['count'] = count
            header = '{} '.format(count)

            ## 每隔一定迭代步数观察效果
            if ((count % 1000 == 0 and count > 0) and (status == SolveStatus.ONGOING)):
                problem.check_sol_status(sol,print_func=print,print_header=header)

            ## 如果最优或者无解，abort
            if status != SolveStatus.ONGOING:
                problem.check_sol_status(sol,print_func=print,print_header=header)
                return status,problem,sol,basis

            ## 限制迭代时长和次数
            if time.time() - start_time > 9.0e2 or count > 1e5:
                print('out of time / iterations.')
                problem.check_sol_status(sol,print_func=print,print_header=header)
                return SolveStatus.OTHER,problem,sol,basis

    def _solve(self,problem,sol,basis):
        ## 直接进入DS迭代
        return self._loop(problem,sol,basis)
            
    def solve(self,A_raw,b_raw,sense_raw,c_raw,l_raw,u_raw):
        '''
        主求解入口
        '''
        self.global_info = {'count':0,'start_time':time.time()}

        ## 读取数据
        A,b,sense,c,l,u = A_raw.copy(),b_raw.copy(),sense_raw.copy(),c_raw.copy(),l_raw.copy(),u_raw.copy()
        n,m = A.shape
        ## 加上逻辑变量，保证A是满秩的；否则会出现数值问题
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

        ## 初始化
        problem = Problem(A,b,c,l,u)
        n,m = A.shape
        idxB = np.arange(m-n,m,1,dtype=int)
        basis = Basis(A)
        basis.init_DSE_weights()
        basis.reset_basis_idx(idxB)
        problem,sol,basis = self._refactorize(problem,sol=None,basis=basis)

        ## 开始求解流程
        status,problem,sol,basis = self._solve(problem,sol,basis)

        ## 对原始变量的后处理，去除增加的逻辑变量
        sol.x = sol.x[:(m-n)]

        return status,sol,basis