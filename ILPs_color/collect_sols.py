import multiprocessing

import pyscipopt as scip
import os
import numpy as np
import gzip,pickle
import argparse
from feature_extract import extract_features


def getSolObjs(m):
    solDicts = m.getSols()
    objs = [m.getSolObjVal(sol) for sol in solDicts]
    vars = [va for va in m.getVars()]
    varTypes = [va.vtype() for va in vars]
    sols = np.zeros((len(solDicts), len(vars)))
    for i in range(len(solDicts)):
        for j in range(len(vars)):
            sols[i, j] = solDicts[i].__getitem__(vars[j])
            if varTypes[j] in ['BINARY','INTEGER'] :
                sols[i,j] = round(sols[i,j]) # round integer values


    return sols,objs

def collect(filepath,saveRoot,nSol,maxTime):

    insName = os.path.basename(filepath)

    # extract MILP graph
    varNames, variableFeatures, constraintFeatures, edgeInds, edgeWeights = extract_features(filepath)
    bpSaveDir = os.path.join(saveRoot, 'bipartites')
    filename = os.path.join(bpSaveDir, f'{insName}.bp')
    with gzip.open(filename, "wb") as f:
        pickle.dump({
            'varNames': varNames,
            'variableFeatures': variableFeatures,
            'constraintFeatures': constraintFeatures,
            'edgeInds': edgeInds,
            'edgeWeights': edgeWeights
        }, f)


    # collect solutions
    m = scip.Model()
    m.hideOutput()
    m.setLogfile(os.path.join(saveRoot, 'logs', insName + '.log'))
    m.readProblem(filepath)

    m.setParams(
        {
            "limits/maxsol": nSol,
            "limits/time": maxTime,
        }
    )

    vars = m.getVars()
    varNames = [va.name  for va in vars]
    varTypes = [va.vtype()  for va in vars]

    varNames = np.array(varNames)
    varTypes = np.array(varTypes)

    m.optimize()

    # get solutions
    sols, objs = getSolObjs(m)
    filename = os.path.join(saveRoot, 'solutions', f'{insName}.sol')
    with gzip.open(filename, "wb") as f:
        pickle.dump({
            'varNames':varNames,
            'varTypes':varTypes,
            'sols': sols,
            'objs': objs,
            'objSense': m.getObjectiveSense()
        }, f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--rootDir', type=str, default=r'data/BPP/train', help='root directory')
    parser.add_argument('--nSol', type=int, default=100, help='number of sols')
    parser.add_argument('--maxTime', type=int, default=300, help='max solving time')
    parser.add_argument('--nWorkers', type=int, default=1, help='number of processes')
    args = parser.parse_args()
    NSOL = args.nSol
    MAX_TIME = args.maxTime
    NWORKER = args.nWorkers
    rootDir = args.rootDir
    os.makedirs(os.path.join(rootDir, 'solutions'), exist_ok=True)
    os.makedirs(os.path.join(rootDir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(rootDir, 'bipartites'), exist_ok=True)
    insDir = os.path.join(rootDir,'instances')
    insNameList = os.listdir(insDir)
    insPathList = [os.path.join(insDir, insName) for insName in insNameList]

    with multiprocessing.Pool(processes=NWORKER) as pool:
        for insPath in insPathList:
            pool.apply_async(collect, (insPath, rootDir,NSOL,MAX_TIME))

        pool.close()
        pool.join()

    print('done')