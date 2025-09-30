import time

import pyscipopt as scip
import numpy as np
import os





def extract_features(filepath):


    m = scip.Model()
    m.hideOutput()
    m.readProblem(filepath)
    conss = m.getConss()
    vars = m.getVars()
    varDict = {}
    nVar = len(vars)
    nCons = len(conss)
    objSense = m.getObjectiveSense()
    varNames = [va.name for va in vars]
    # varTypes = [va.vtype() for va in vars]
    # variable features
    isInteger = np.zeros(nVar)
    isContinous = np.zeros(nVar)
    hasLb = np.zeros(nVar)
    hasUb = np.zeros(nVar)
    lbs = np.zeros(nVar)
    ubs = np.zeros(nVar)
    objCoeffs = np.zeros(nVar)

    # cons features
    bias = np.zeros(nCons)
    consTypes = np.zeros(nCons)

    for ind,va in enumerate(vars):

        varDict[va.name] = ind

        if va.vtype() == 'BINARY' or va.vtype() == 'INTEGER':
            isInteger[ind] = 1
        elif  va.vtype() == 'CONTINUOUS':
            isContinous[ind] = 1
        else:
            raise NotImplementedError

        objCoeffs[ind] = va.getObj() if objSense == 'minimize' else - va.getObj() if objSense == 'maximize' else None

        lb = va.getLbLocal()
        ub = va.getUbLocal()

        if lb == -1e+20:
            lbs[ind] = 0
            hasLb[ind] = 0
        else:
            lbs[ind] = lb
            hasLb[ind] = 1

        if ub == 1e+20:
            ubs[ind] = 0
            hasUb[ind] = 0
        else:
            ubs[ind] = ub
            hasUb[ind] = 1


    edgeInds = []
    edgeWeights = []
    for ind,cons in enumerate(conss):

        lhs = m.getLhs(cons)
        rhs = m.getRhs(cons)

        if lhs==rhs:
            consTypes[ind] = 0 # equal
            bias[ind] = lhs
        elif lhs == -1e+20:
            consTypes[ind] = -1 # less than
            bias[ind] = rhs
        elif rhs == 1e+20:
            consTypes[ind] = 1 # greater than
            bias[ind] = lhs

        consCoffs = m.getValsLinear(cons)

        for varname, weight in consCoffs.items():
            edgeInds.append([ ind, varDict[varname]]) # cons ID -> var ID
            edgeWeights.append(weight)

    edgeInds = np.array(edgeInds) # cons id -> var id
    edgeWeights = np.array(edgeWeights)

    variableFeatures = np.stack([isInteger,isContinous,hasLb,hasUb,lbs,ubs,objCoeffs],axis=-1)
    constraintFeatures = np.stack([consTypes,bias],axis=-1)

    return varNames,variableFeatures,constraintFeatures,edgeInds,edgeWeights

if __name__ == '__main__':

    # verification

    filepath = 'ins.lp'
    # create a instance
    m = scip.Model()
    # add variable
    x0 = m.addVar(vtype='B')
    x1 = m.addVar(vtype='B')
    x2 = m.addVar(vtype='I',lb=None,ub=8)
    x3 = m.addVar(vtype='C',lb=3)
    # add cons
    c1 = m.addCons(x1 +1.1* x2 <= 2)
    c2 = m.addCons(1.1*x1 + 1.3*x3 >= 3)
    c3 = m.addCons(1.02*x2 + 1.2*x3 == 4)
    c4 = m.addCons(x0 + x1 == 1)
    # add obj
    m.setObjective(-0.5*x0 + x1 + 0.2*x2 + 0.3*x3, sense = 'minimize')

    m.writeProblem('test_ins.lp')

    variableFeatures,constraintFeatures,edgeInds,edgeWeights = extract_features('test_ins.lp')
    # st = time.time()
    # variableFeatures, constraintFeatures, edgeInds, edgeWeights = extract_features('square47.mps.gz')
    # cost = time.time() - st

    print('done')
