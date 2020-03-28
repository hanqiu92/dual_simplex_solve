import numpy as np
import math
from scipy.sparse import dok_matrix

INF = 1e16

def read_mps(fname):
    with open(fname,'r+') as f:
        lines = f.readlines()

    line_type = 'NONE'

    row_key,row_idx = {},0
    col_key,col_idx = {},0
    nonzero_cnt = 0

    A,b,sense,c,l,u = dict(),dict(),dict(),dict(),dict(),dict()

    for line in lines:
        line_strip = line.strip()
        if (line[0] == '*') or line.isspace():
            continue

        if len(line_strip.split()) == 1:
            if line_strip in ('ROWS','COLUMNS','RHS','RANGES','BOUNDS'):
                line_type = line_strip
                continue
            elif line_strip == 'ENDATA':
                break
            else:
                line_type = 'NONE'
                print('Unknown line type! Tag: {}'.format(line_strip))
                continue
        
        num_fields = min(int(math.ceil((len(line) - 14) / 25)),2)

        if line_type == 'ROWS':
            row_type = line[1:3].strip()
            row_name = line[4:14].strip()
            if row_type == 'N':
                obj_row_name = row_name
            else:
                row_key[row_name] = row_idx
                if row_type == 'E':
                    sense[row_idx] = 0
                elif row_type == 'G':
                    sense[row_idx] = 1
                elif row_type == 'L':
                    sense[row_idx] = -1
                else:
                    print(line)
                row_idx += 1

        elif line_type == 'COLUMNS':
            col_type = line[1:3].strip()
            col_name = line[4:14].strip()
            if col_name not in col_key:
                col_key[col_name] = col_idx
                col_idx += 1
            for i in range(num_fields):
                idx = 14 + i * 25
                row_name = line[idx:(idx+10)].strip()
                value = line[(idx+10):(idx+22)].strip()
                if value != '':
                    if row_name == obj_row_name:
                        c[col_key[col_name]] = float(value)
                    else:
                        if float(value) != 0:
                            A[(row_key[row_name],col_key[col_name])] = float(value)
                            nonzero_cnt += 1


        elif line_type == 'RHS':
            rhs_type = line[1:3].strip()
            rhs_name = line[4:14].strip()
            for i in range(num_fields):
                idx = 14 + i * 25
                row_name = line[idx:(idx+10)].strip()
                row_value = line[(idx+10):(idx+22)].strip()
                if row_name != obj_row_name and row_value != '':
                    b[row_key[row_name]] = float(row_value)

        elif line_type == 'RANGES':
            range_type = line[1:3].strip()
            range_name = line[4:14].strip()
            for i in range(num_fields):
                idx = 14 + i * 25
                row_name = line[idx:(idx+10)].strip()
                row_value = line[(idx+10):(idx+22)].strip()
                if row_name != obj_row_name and row_value != '':
                    row_idx_ = row_key[row_name]
                    row_sense = sense[row_idx_]
                    row_value = float(row_value)
                    if (row_sense == -1) or ((row_sense == 0) and (row_value < 0)):
                        ## le or eq < 0
                        bnds = (b[row_idx_]-abs(row_value),b[row_idx_])
                    elif (row_sense == 1) or ((row_sense == 0) and (row_value >= 0)):
                        ## ge or eq >= 0
                        bnds = (b[row_idx_],b[row_idx_]+abs(row_value))

                    ## turn this row into a varible!
                    ##       b_l <= a^T x <= b_u
                    ## =>    a^T x - x' = 0, b_l <= x' <= b_u
                    col_key[row_name] = col_idx
                    c[col_idx],l[col_idx],u[col_idx] = 0,bnds[0],bnds[1]
                    b[row_idx_],sense[row_idx_] = 0,0
                    A[(row_idx_,col_idx)] = -1
                    col_idx += 1

        elif line_type == 'BOUNDS':
            bound_type = line[1:3].strip()
            bound_name = line[4:14].strip()
            col_name = line[14:24].strip()
            bound_value = line[24:36].strip()
            if bound_type == 'UP':
                u[col_key[col_name]] = float(bound_value)
            elif bound_type == 'LO':
                l[col_key[col_name]] = float(bound_value)
            elif bound_type == 'FX':
                u[col_key[col_name]] = float(bound_value)
                l[col_key[col_name]] = float(bound_value)
            elif bound_type == 'FR':
                l[col_key[col_name]] = -INF
            elif bound_type == 'PL':
                pass
            elif bound_type == 'MI':
                u[col_key[col_name]] = 0
                l[col_key[col_name]] = -INF
            else:
                ## not handle yet
                print(line)
                pass

    
    n,m = row_idx,col_idx
    return A,b,sense,c,l,u,m,n,row_key,col_key

def dict_to_sp_mat(dict_,n,m):
    mat = dok_matrix((n,m))
    for key,value in dict_.items():
        mat[key[0],key[1]] = value
    mat = mat.tocsc()
    return mat

def dict_to_dense_vec(dict_,size,default=0):
    vec = default * np.ones((size,))
    for key,value in dict_.items():
        vec[key] = value
    return vec

def dicts_to_computable(A_dict,b_dict,sense_dict,c_dict,l_dict,u_dict,m,n):
    A = dict_to_sp_mat(A_dict,n,m)
    b = dict_to_dense_vec(b_dict,n)
    sense = dict_to_dense_vec(sense_dict,n)
    c = dict_to_dense_vec(c_dict,m)
    l = dict_to_dense_vec(l_dict,m)
    u = dict_to_dense_vec(u_dict,m,default=INF)
    return A,b,sense,c,l,u

import pulp
from pulp.solvers import PULP_CBC_CMD
def solve_pulp(A_dict,b_dict,sense_dict,c_dict,l_dict,u_dict,m,n,msg=0):
    solver = PULP_CBC_CMD(msg=msg)
    model = pulp.LpProblem("test",pulp.LpMinimize)
    model.solver = solver

    A_by_row = dict([(j,dict()) for j in range(n)])
    for key,value in A_dict.items():
        A_by_row[key[0]][key[1]] = value

    x_pulp = dict([(i,pulp.LpVariable(str(i),lowBound=l_dict.get(i,0),upBound=u_dict.get(i,INF))) for i in range(m)])
    cons_pulp = dict()
    for j in range(n):
        cons_pulp[j] = pulp.LpConstraint(pulp.LpAffineExpression(
                [(x_pulp[i],value) for i,value in A_by_row[j].items()],
                constant=-b_dict.get(j,0)),
            sense=sense_dict.get(j,0))
        model += cons_pulp[j]
    model.objective = pulp.LpAffineExpression([(x_pulp[key],value) for key,value in c_dict.items()])
    model.solve()

    x = np.zeros((m,))
    for i in range(m):
        x[i] = x_pulp[i].value()
    x[np.isnan(x)] = 0
    lam = np.zeros((n,))
    for j in range(n):
        lam[j] = cons_pulp[j].pi
    lam[np.isnan(lam)] = 0
    return x,lam,model.status

class Evaluator(object):
    def __init__(self,A=None,b=None,sense=None,c=None,l=None,u=None):
        if A is not None:
            self.reset(A,b,sense,c,l,u)

    def reset(self,A,b,sense,c,l,u):
        self.A = A
        self.b = b
        self.sense = sense
        self.c = c
        self.l = l
        self.u = u

    def eval_con_inf(self,x):
        err = self.A.dot(x) - self.b
        return ( np.sum(err[(err >= 0) & (self.sense <= 0)]) - \
                 np.sum(err[(err <= 0) & (self.sense >= 0)]) )

    def eval_primal_inf(self,x):
        return ( np.sum(np.maximum(self.l - x,0)) + np.sum(np.maximum(x - self.u,0)) )

    def eval_primal_obj(self,x):
        return np.dot(self.c,x)

    def eval_str(self,x):
        return ('con inf={:.4e},var inf={:.4e},obj={:.4e}.'.format(
            self.eval_con_inf(x),self.eval_primal_inf(x),self.eval_primal_obj(x)
        ))


import time
import traceback
import glob
np.set_printoptions(suppress=False,precision=4)

def test(solve_func,max_problem_size=1e20,model_fnames=None,output_fname=None,random_seed=None):
    '''
    测试函数，各参数的定义如下：
    solve_func对应所实现的DS求解流程入口
    problem_size代表最大可容许的问题矩阵A的非零元素个数，用于约束测试问题的难度
    model_fnames对应一个测试问题路径的list，进一步提供了只对部分问题进行测试的选项
    output_fname对应结果输出路径，可输出PuLP和实现的DS流程对各测试问题的求解结果和时间，用于后续数据分析
    random_seed可指定随机种子，用于在存在随机性的求解流程（例如随机扰动）中固定随机项，保证多次调用的结果的稳定性
    '''
    if output_fname is not None:
        ## 打开输出文件并写入表头
        f = open(output_fname,'w+')
        f.write('model name\tsize\ttime solver\tobj solver\ttime pulp\tobj pulp\tobj matched?\n')
    
    if random_seed is None:
        ## 用当前时间作为随机种子
        random_seed = int(time.time())
    print('random seed = {}.'.format(random_seed))
    
    evaluator = Evaluator()
    if model_fnames is None:
        ## 默认读取所有测试问题
        model_fnames = sorted(glob.glob('netlib/*.SIF'))
    for fname in model_fnames:
        model_name = fname.split('/')[-1].split('.')[0]

        ## 读取测试问题
        A_dict,b_dict,sense_dict,c_dict,l_dict,u_dict,m,n,row_key,col_key = read_mps(fname)
        A,b,sense,c,l,u = dicts_to_computable(A_dict,b_dict,sense_dict,c_dict,l_dict,u_dict,m,n)

        ## 根据测试问题大小进行筛选
        if A.nnz < max_problem_size:
            evaluator.reset(A,b,sense,c,l,u)
            size = (A.shape,A.nnz)
            print('\nProblem name: {}, size: ({},{}).'.format(model_name,A.shape,A.nnz))
            
            time_pulp,time_ds,obj_pulp,obj_ds = np.nan,np.nan,np.nan,np.nan
            bool_match = False
            try:
                ## 调用PuLP求解
                tt = time.time()
                print("[----Launch PuLP----]")
                x_pulp,lam_pulp,status_pulp = solve_pulp(A_dict,b_dict,sense_dict,c_dict,l_dict,u_dict,
                                                         m,n,msg=1)
                time_pulp = time.time() - tt

                ## 调用实现的DS流程求解
                np.random.seed(random_seed)
                tt = time.time()
                print("[----Launch Solver----]")
                status_ds,sol_ds,basis_ds = solve_func(A,b,sense,c,l,u)
                x_ds = sol_ds.x
                time_ds = time.time() - tt
                
                ## 比较求解结果
                obj_pulp = evaluator.eval_primal_obj(x_pulp)
                obj_ds = evaluator.eval_primal_obj(x_ds)
                print("[----Begin evaluation----]")
                print(f"PuLP Eval: {evaluator.eval_str(x_pulp)} Status: {status_pulp}.")
                print(f"Solver Eval: {evaluator.eval_str(x_ds)} Status: {status_ds}.")
                print(f"Elapsed time: PuLP = {time_pulp:.3f}, Solver = {time_ds:.3f}.")
                bool_match = np.abs(obj_ds - obj_pulp) <= 1e-4 * np.abs(obj_pulp)
                print(f"Two solvers {'match' if bool_match else 'mismatch'}: "
                      f"PuLP = {obj_pulp:.4e}, Solver = {obj_ds:.4e}.")

            except Exception as e:
                print(repr(e))
                print(traceback.print_exc())
                
            if output_fname is not None:
                ## 输出求解结果
                f.write(f"{model_name}\t{size}\t{time_ds:.3f}\t{obj_ds:.4e}\t"
                        f"{time_pulp:.3f}\t{obj_pulp:.4e}\t{bool_match}\n")

    if output_fname is not None:
        f.close()