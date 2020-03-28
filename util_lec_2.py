import numpy as np
import scipy.sparse as sp
import time
from enum import Enum,unique

## 一些与数值误差相关的常数的定义
INF = 1e16
PRIMAL_TOL = 1e-7
PRIMAL_RELA_TOL = 1e-9
DUAL_TOL = 1e-7
CON_TOL = 1e-5
PIVOT_TOL = 1e-5
REMOVE_TOL = 1e-14
ZERO_TOL = 1e-12


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