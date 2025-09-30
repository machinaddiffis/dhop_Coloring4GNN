from ortools.linear_solver.python import model_builder
from ortools.linear_solver.python import model_builder_helper as mbh
import numpy as np
import networkx as nx
import gzip
import os
# import torch
import pickle
import math
import time
import os
import psutil
import torch
import random
from ortools.pdlp import solve_log_pb2
from ortools.pdlp import solvers_pb2
from ortools.pdlp.python import pdlp
import argparse
pcs=psutil.Process()
import multiprocessing
# import pyscipopt as scp
# local test
# fdir = "./bdry2.mps"

def solve_get_sol2(model,sol1,dual1):
    ptc=model.export_to_proto()
    print("hello")
    mdl = get_prob(ptc)
    print("hello")
    params = solvers_pb2.PrimalDualHybridGradientParams()
    opt_criteria = params.termination_criteria.simple_optimality_criteria
    opt_criteria.eps_optimal_relative = 1.0e-4
    result = pdlp.primal_dual_hybrid_gradient(
        mdl, params
    )
    for e in dir(mdl):
        print(e)
    ps=result.primal_solution
    ds=result.dual_solution
    vns=mdl.variable_names
    cns=mdl.constraint_names
    x={}
    y={}
    for idx,v in enumerate(vns):
        sol1.append(ps[idx])
    for idx,c in enumerate(cns):
        dual1.append(ds[idx])
    print(ps)
    print(ds)
    return sol1,dual1

def get_prob(prt,realx_int=False,include_name=True):
    return pdlp.qp_from_mpmodel_proto(prt,realx_int,include_name)

def getFeat(model, solve=True, use_pos=False):
    data = {}
    nc = model.num_constraints
    nv = model.num_variables
    var_names=[]
    con_names=[]
    b_coef1=model.get_linear_constraint_lower_bounds().values
    b_coef2=model.get_linear_constraint_upper_bounds().values
    b_coef=[]
    for i in range(len(b_coef1)):
        if np.isinf(b_coef1[i]):
            b_coef.append(b_coef2[i])
        else:
            b_coef.append(b_coef1[i])
    solver = model_builder.ModelSolver("PDLP")

    solver.enable_output(True)

    vs = model.get_variables()
    for v in vs:
        var_names.append(v.name)

    v_map = {}
    v_feat = []

    inf = float('inf')
    c_obj = []
    for v in vs:
        vname = v.name
        vidx = v.index
        lb = max(v.lower_bound, -1e+5)
        ub = min(v.upper_bound, 1e+5)
        c = v.objective_coefficient
        if lb == -inf:
            lb = -1e+5
        if ub == inf:
            ub = 1e+5
        v_map[vname] = vidx
        v_feat.append([lb, ub, c])
        c_obj.append(c)
    # need to convert v_feat to either numpy or torch tensor

    c_map = {}
    c_feat = []

    # nvar X ncon
    e_indx = [[], []]
    e_val = []

    cs = model.get_linear_constraints()
    
    for c in cs:
        con_names.append(c.name)
    exprs = model.get_linear_constraint_expressions()
    for idx, c in enumerate(cs):
        act_idx = c.index
        cn = c.name
        c_map[cn] = idx
        lb = c.lower_bound
        ub = c.upper_bound
        tmp = [0, 0, 0, None]
        # leq eq geq lb ub
        if lb <= -1e+20 and ub < 1e+20:
            tmp[0] = 1.0
            tmp[3] = ub
            # tmp[4] = ub
        elif ub >= 1e+20 and lb > -1e+20:
            tmp[2] = 1.0
            tmp[3] = lb
            # tmp[4] = 1e+20
        elif lb == ub:
            tmp[1] = 1.0
            tmp[3] = lb
            # tmp[4] = lb
        elif ub != lb:
            print('ERROR::unhandled constraint')
            print(lb, "<=", cn, "<=", ub)
            quit()
        c_feat.append(tmp)
        expr = exprs[act_idx]
        tm = expr.variable_indices()
        coef = expr.coeffs
        for ii in range(len(tm)):
            e_indx[0].append(tm[ii])
            e_indx[1].append(act_idx)
            e_val.append(coef[ii])
    print(f'processed file with {nv}vars, {nc}cons, {len(e_val)}nnz')

    # add "pos emb" for each
    if use_pos:
        # -- v
        eps = 0.5
        dim = 16
        for i in range(len(v_feat)):
            tmp = [0] * (dim * len(v_feat[i]))
            for k in range(len(v_feat[i])):
                for j in range(dim // 2):
                    tmp[dim * k + 2 * j] = math.sin(eps * v_feat[i][k] / math.pow(10000, 2 * j / dim))
            v_feat[i] = tmp
        # -- c
        eps = 0.5
        dim = 16
        for i in range(len(c_feat)):
            tmp = [0] * (dim * len(c_feat[i]))
            for k in range(len(c_feat[i])):
                for j in range(dim // 2):
                    tmp[dim * k + 2 * j] = math.sin(eps * c_feat[i][k] / math.pow(10000, 2 * j / dim))
            c_feat[i] = tmp
    dual_map = []
    sol_map = []
    sol_map,dual_map = solve_get_sol2(model,sol_map,dual_map)

    data["edge_index"] = torch.tensor(e_indx, dtype=torch.int32).long()
    data["edge_weight"] = torch.tensor(e_val, dtype=torch.float32)
    data["con_feat"] = torch.tensor(c_feat, dtype=torch.float32)
    data["var_feat"] = torch.tensor(v_feat, dtype=torch.float32)
    data["b"] = torch.tensor(np.array(b_coef).reshape(data["con_feat"].shape[0], 1), dtype=torch.float32)
    data["c"] = torch.tensor(np.array(c_obj).reshape(data["var_feat"].shape[0], 1), dtype=torch.float32)
    data["label"] = torch.tensor(sol_map, dtype=torch.float32)
    data["dual"] = torch.tensor(dual_map, dtype=torch.float32)
    data["names"]=[var_names,con_names]
    return data

def generate_pagerank_mps_dataset(num_nodes, degree, damping_factor):
    model = model_builder.ModelBuilder()
    print('Generating Graph',pcs.memory_info().rss)
    graph = nx.barabasi_albert_graph(num_nodes, degree)
    print('Graph generated',pcs.memory_info().rss)

    adjacency_matrix = nx.adjacency_matrix(graph)
    print('Adj Mat generated',pcs.memory_info().rss)

    stochastic_matrix_prime = adjacency_matrix / adjacency_matrix.sum(axis=0)
    adjacency_matrix=None
    print('Matrix Normalized ',pcs.memory_info().rss)

    lp_coefficients = damping_factor * stochastic_matrix_prime
    print('Coefficient Finished',pcs.memory_info().rss)
    
    lp_coefficients=lp_coefficients.tocsr()
    print(type(lp_coefficients))
    
    stt=lp_coefficients.indptr
    idx=lp_coefficients.indices
    val=lp_coefficients.data

    var_maps={}
    for i in range(num_nodes):
        if i%100==0:
            print(f'adding var: --------x_{i}/{num_nodes}---------',pcs.memory_info().rss)
        var_maps[f'x_{i}']=model.new_var(name=f'x_{i}',lb=0.0,ub=1e+20,is_integer=False)
    print('Finished adding Vars')
    
    #add cons
    rhs=-(1 - damping_factor) / num_nodes
    for i in range(num_nodes):
        if i%100==0:
            print(f'adding cons PRC: --------{i}/{num_nodes}---------')
        st=stt[i]
        ed=stt[i+1]
        terms=[]
        coeffs=[]
        flag=None
        expr1 = 0.0
        for ji,j in enumerate(idx[st:ed]):
            terms.append(var_maps[f'x_{j}'])
            if j==i:
                flag=i
            coeffs.append(val[st+ji])
        if flag is None:
            terms.append(var_maps[f'x_{i}'])
            coeffs.append(-1.0)
        else:
            coeffs[flag]-=1.0
        for k in range(len(coeffs)):
            expr1 += coeffs[k]*terms[k]
        # model.add(model_builder.LinearExpr().weighted_sum(terms,coeffs) <= rhs,name=f'PRC_{i}')
        model.add(expr1 <= rhs,name=f'PRC_{i}')
    exps=[]
    expr = 0.0
    print('Finished adding Cons  1')
    for i in range(num_nodes):
        # exps.append(var_maps[f'x_{i}'])
        expr += var_maps[f'x_{i}']
    # model.add(model_builder.LinearExpr().sum(exps) == 1.0,name=f'SING')
    model.add(expr == 1.0,name=f'SING')
    print('Finished adding Cons  2')
    return model

def train_data(train_num, num_nodes, degree, damping_factor, ident="Train"):
    model = generate_pagerank_mps_dataset(num_nodes, degree, damping_factor)
    data = getFeat(model, True)

    fout = gzip.open(f'../data/{ident}/packingdata{train_num}.pkl','w')
    pickle.dump(data,fout)
    fout.close()


def gen_files(n_files, num_nodes, degree, damping_factor, nworker = 8):
    if not os.path.isdir('../data/train'):
        os.mkdir('../data/train')
    if not os.path.isdir('../data/valid'):
        os.mkdir('../data/valid')
    pool = multiprocessing.Pool(processes = nworker)
    for i in range(n_files[0]):
        pool.apply_async(train_data, args=(i, num_nodes, degree, damping_factor, "train"))
        # train_data(i, num_nodes, degree, damping_factor, ident="train")
    for i in range(n_files[1]):
        pool.apply_async(train_data, args=(i, num_nodes, degree, damping_factor, "valid"))
        # train_data(i, num_nodes, degree, damping_factor, ident="valid")
    pool.close()
    pool.join()

    
def gen_test(n_files, num_nodes, degree, damping_factor, nworker = 8):
    if not os.path.isdir('../data/test'):
        os.mkdir('../data/test')
    pool = multiprocessing.Pool(processes = nworker)
    for i in range(n_files[2]):
        pool.apply_async(train_data, args=(i, num_nodes, degree, damping_factor, "test"))
        # train_data(i, num_nodes, degree, damping_factor, ident="test")
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--nnodes", type=int, help="number of nodes", default=10000)
    parser.add_argument("-d","--degree", type=int, help="number of degree", default=3)
    parser.add_argument("-p","--damping_factor", type=float, help="damping factor", default=0.85)
    parser.add_argument("-t","--ntrain", type=int, help="number of training ins", default=90)
    parser.add_argument("-v","--nvalid", type=int, help="number of valid ins", default=10)
    parser.add_argument("-s","--ntest", type=int, help="number of test ins", default=10)
    args = parser.parse_args()


    num_nodes = args.nnodes
    degree = args.degree
    damping_factor = args.damping_factor
    t1 = time.time()
    n_files = [args.ntrain,args.nvalid,args.ntest]
    gen_files(n_files, num_nodes, degree, damping_factor)
    gen_test(n_files, num_nodes*5, degree, damping_factor)
    t2 = time.time()
    print('Instance generation finished in',t2-t1,'s')