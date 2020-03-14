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
    lam = np.zeros((n,))
    for j in range(n):
        lam[j] = cons_pulp[j].pi
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
