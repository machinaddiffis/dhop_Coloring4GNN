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




import logging

from absl import app
from absl import flags

from google.protobuf import text_format
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver import pywraplp
import math
from ortools.linear_solver.python import model_builder
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.pdlp import solve_log_pb2
from ortools.pdlp import solvers_pb2
import numpy as np
from ortools.pdlp.python import pdlp

import time
import gurobipy
# import pyscipopt as scp
# local test
# fdir = "./bdry2.mps"


CHECK_FEAS = True


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

def getFeat(i,model, solve=True, use_pos=False):
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

    if CHECK_FEAS:
        model.write_to_mps_file(f'../tmp/t{i}.mps')
        gbm = gurobipy.read(f'../tmp/t{i}.mps')
        gbm.optimize()
        os.remove(f'../tmp/t{i}.mps')
        if gbm.Status != gurobipy.GRB.OPTIMAL:
            return None


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

def BuildMipForLoadBalancing(problem):
    print('Building...')
    model = model_builder.ModelBuilder()

    num_workloads = len(problem.workload)
    num_workers = len(problem.worker)

    # reserved_capacity_vars[s][t] contains the index of a continuous variable
    # denoting the capacity reserved on worker t for handling workload s.
    reserved_capacity_vars = [[None] * num_workers for _ in range(num_workloads)]
    for s in range(num_workloads):
        for t in range(num_workers):
            vname = f'reserved_capacity_{s}_{t}'
            reserved_capacity_vars[s][t] = model.new_var(name=vname, lb=0.0, ub=0.0, is_integer=False)
        # capacity can be reserved only on allowed workers.
        for t in problem.workload[s].allowed_workers:
            reserved_capacity_vars[s][t].upper_bound = problem.worker[t].capacity
    print(f'Finished adding reserved_capacity_s_t')

    # worker_used_vars[t] contains the index of a binary variable denoting whether
    # worker t is used in the solution.
    worker_used_vars = [None] * num_workers
    for t in range(num_workers):
        vname = f'worker_used_{t}'
        worker_used_vars[t] = model.new_var(name=vname, lb=0.0, ub=1.0, is_integer=False)
        worker_used_vars[t].objective_coefficient = problem.worker[t].cost
    print(f'Finished adding worker_used_t')

    # Capacity can be reserved only if a node is used.
    # For s, t: reserved_capacity_vars[s][t] <= M * worker_used_vars[t]
    # where M = max(worker[t].capacity, workload[s].load)
    for s in range(num_workloads):
        for t in range(num_workers):
            cname = f'worker_used_ct_{s}_{t}'
            exps = []
            coeffs = []
            exps.append(reserved_capacity_vars[s][t])
            coeffs.append(-1)
            exps.append(worker_used_vars[t])
            M = max(problem.worker[t].capacity, problem.workload[s].load)
            coeffs.append(M)
            expr = 0.0
            for i in range(len(exps)):
                expr += exps[i] * coeffs[i]
            # model.add(model_builder.LinearExpr().weighted_sum(exps, coeffs) >= 0.0, name=cname)
            model.add(expr >= 0.0,name=cname)

    print(f'Finished adding worker_used_ct_s_t')

    # Cannot reserve more capacity than available.
    # For t: sum_s reserved_capacity_vars[s][t] <= worker[t].capacity
    for t in range(num_workers):
        cname = f'worker_capacity_ct_{t}'
        exps = []
        coeffs = []
        expr = 0.0
        for s in range(num_workloads):
            # exps.append(reserved_capacity_vars[s][t])
            # coeffs.append(1)
            expr += reserved_capacity_vars[s][t]
        # model.add(model_builder.LinearExpr().weighted_sum(exps, coeffs) <= problem.worker[t].capacity, name=cname)
        model.add(expr <= problem.worker[t].capacity,name=cname)
    print(f'Finished adding worker_capacity_ct_t')

    # There must be sufficient capacity for each workload in the scenario where
    # any one of the allowed workers is unavailable.
    # For s, t: sum_{t' != t} reserved_capacity_vars[s][t'] >= workload[s].load
    for s in range(num_workloads):
        allowed_workers = sorted(set(problem.workload[s].allowed_workers))
        assert len(allowed_workers) > 1
        for unavailable_t in allowed_workers:
            cname = f'workload_ct_{s}_failure_{unavailable_t}'
            exps = []
            coeffs = []
            expr = 0.0
            for t in allowed_workers:
                if t == unavailable_t:
                    continue
                # exps.append(reserved_capacity_vars[s][t])
                # coeffs.append(1)
                expr += reserved_capacity_vars[s][t]
            # model.add(model_builder.LinearExpr().weighted_sum(exps, coeffs) >= problem.workload[s].load, name=cname)
            model.add(expr >= problem.workload[s].load, name=cname)

    print(f'Finished adding workload_ct_s_failure_unavailable_t')

    return model


class wk:
    def __init__(self, cap, cost):
        self.capacity = cap
        self.cost = cost


class wl:
    def __init__(self, load):
        self.load = load
        self.allowed_workers = []


class lbP:
    def __init__(self):
        self.worker = []
        self.workload = []


def GenerateLoadBalancingProblem(worker_parameter, workload_parameter,random_seed=0):
    random.seed(random_seed)
    problem = lbP()

    # The index of the WorkerParameters that generated this worker.
    worker_group = []

    print(worker_parameter)
    for (group, worker_param) in enumerate(worker_parameter):
        num_workers = random.randint(worker_param["i_min"], worker_param["i_max"])
        for _ in range(num_workers):
            worker = wk(random.uniform(worker_param["capacity_min"], worker_param["capacity_max"]), worker_param["cost"])
            worker_group.append(group)
            problem.worker.append(worker)

    for workload_param in workload_parameter:
        num_workloads = random.randint(workload_param["i_min"], workload_param["i_max"])
        assert len(workload_param["allowed_worker_probability"]) == len(worker_parameter)
        for _ in range(num_workloads):
            workload = wl(random.uniform(workload_param["load_min"], workload_param["load_max"]))
            for worker_index in range(len(problem.worker)):
                if random.random() < workload_param["allowed_worker_probability"][worker_group[worker_index]]:
                    workload.allowed_workers.append(worker_index)

            problem.workload.append(workload)

    return problem

def MPModelProtoToMPS(model_proto: linear_solver_pb2.MPModelProto):
    nnz = sum(len(c.var_index) for c in model_proto.constraint)
    logging.info('# vars = %d, # cons = %d, # nz = %d', len(model_proto.variable),
                 len(model_proto.constraint), nnz)
    model_mps = pywraplp.ExportModelAsMpsFormat(model_proto)
    return model_mps


def BuildRandomizedModels(worker_parameter, workload_parameter,random_seed):
    print('Generating')
    problem = GenerateLoadBalancingProblem(worker_parameter, workload_parameter,random_seed)
    model = BuildMipForLoadBalancing(problem)
    nc = model.num_constraints
    nv = model.num_variables
    print(f'Generated model with {nv}vars and {nc}cons')
    return model

def train_data(train_num, worker_parameter, workload_parameter, ident="Train"):
    data = None
    offset = 0
    while True:
        model = BuildRandomizedModels(worker_parameter, workload_parameter,train_num)
        data = getFeat(train_num,model, True)
        if data is not None:
            break
        offset += 1
        train_num *= 100 + offset

    fout = gzip.open(f'../data/lb/{ident}/packingdata{train_num}.pkl','w')
    pickle.dump(data,fout)
    fout.close()


def gen_files(n_files, worker_parameter, workload_parameter, nworker = 8):
    if not os.path.isdir('../data/lb/train'):
        os.mkdir('../data/lb/train')
    if not os.path.isdir('../data/lb/valid'):
        os.mkdir('../data/lb/valid')
    pool = multiprocessing.Pool(processes = nworker)
    for i in range(n_files[0]):
        train_data(i, worker_parameter, workload_parameter, "train")
    for i in range(n_files[1]):
        train_data(i, worker_parameter, workload_parameter, "valid")
    return()
    for i in range(n_files[0]):
        pool.apply_async(train_data, args=(i, worker_parameter, workload_parameter, "train"))
        # train_data(i, num_nodes, degree, damping_factor, ident="train")
    for i in range(n_files[1]):
        pool.apply_async(train_data, args=(i, worker_parameter, workload_parameter, "valid"))
        # train_data(i, num_nodes, degree, damping_factor, ident="valid")
    pool.close()
    pool.join()

    
def gen_test(n_files, worker_parameter, workload_parameter, nworker = 8):
    if not os.path.isdir('../data/lb/test'):
        os.mkdir('../data/lb/test')
    for i in range(n_files[2]):
        train_data(i, worker_parameter, workload_parameter, "test")
    return 
    pool = multiprocessing.Pool(processes = nworker)
    for i in range(n_files[2]):
        pool.apply_async(train_data, args=(i, worker_parameter, workload_parameter, "test"))
        # train_data(i, num_nodes, degree, damping_factor, ident="test")
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--multiplier", type=int, help="size multiplier", default=1)
    parser.add_argument("--multiplier2", type=int, help="size multiplier for test", default=1.2)
    parser.add_argument("--loadmin", type=str, help="load min, seperate by ,", default="1.0,5.0")
    parser.add_argument("--loadmax", type=str, help="load max, seperate by ,", default="5.0,10.0")
    parser.add_argument("--prob1", type=str, help="allowed_worker_probability 1", default="0.15,0.3")
    parser.add_argument("--prob2", type=str, help="allowed_worker_probability 2", default="0.3,0.15")
    parser.add_argument("-t","--ntrain", type=int, help="number of training ins", default=190)
    parser.add_argument("-v","--nvalid", type=int, help="number of valid ins", default=10)
    parser.add_argument("-s","--ntest", type=int, help="number of test ins", default=10)
    args = parser.parse_args()

    lmin = [float(x) for x in args.loadmin.split(',')]
    lmax = [float(x) for x in args.loadmax.split(',')]
    p1 = [float(x) for x in args.prob1.split(',')]
    p2 = [float(x) for x in args.prob2.split(',')]

    worker_parameter = [
        {"i_min": 100 * args.multiplier, "i_max": 100 * args.multiplier, "capacity_min": 0.5, "capacity_max": 0.8, "cost": 1},
        {"i_min": 100 * args.multiplier, "i_max": 100 * args.multiplier, "capacity_min": 0.4, "capacity_max": 0.8, "cost": 1}
    ]

    workload_parameter = [
        {"i_min": 20 * args.multiplier, "i_max": 20 * args.multiplier, "load_min": 1.0, "load_max": 5.0, "allowed_worker_probability": [0.15, 0.3]},
        {"i_min": 5 * args.multiplier, "i_max": 5 * args.multiplier, "load_min": 5.0, "load_max": 10.0, "allowed_worker_probability": [0.3, 0.15]}
    ]

    
    worker_parameter2 = [
        {"i_min": int(round(100 * args.multiplier2)), "i_max": int(round(100 * args.multiplier2)), "capacity_min": 0.5, "capacity_max": 0.8, "cost": 1},
        {"i_min": int(round(100 * args.multiplier2)), "i_max": int(round(100 * args.multiplier2)), "capacity_min": 0.4, "capacity_max": 0.8, "cost": 1}
    ]

    workload_parameter2 = [
        {"i_min": int(round(20 * args.multiplier2)), "i_max": int(round(20 * args.multiplier2)), "load_min": 1.0, "load_max": 5.0, "allowed_worker_probability": [0.15, 0.3]},
        {"i_min": int(round(5 * args.multiplier2)), "i_max": int(round(5 * args.multiplier2)), "load_min": 5.0, "load_max": 10.0, "allowed_worker_probability": [0.3, 0.15]}
    ]

    if not os.path.isdir('../data/lb'):
        os.mkdir('../data/lb')

    t1 = time.time()
    n_files = [args.ntrain,args.nvalid,args.ntest]
    # gen_files(n_files, worker_parameter, workload_parameter)
    gen_test(n_files, worker_parameter2, workload_parameter2)
    t2 = time.time()
    print('Instance generation finished in',t2-t1,'s')