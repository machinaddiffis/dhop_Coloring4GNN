import pyscipopt as scip
import numpy as np
import random
import os

def gen_bin_packing_ins(capacity=100,largeRatio=0.3,largeLB=0.8,largeUB=1.0,smallLB=0.1,smallUB=0.3,nItems=10):

    largeItemTypes = list(range(int(largeLB*capacity)+1,int(largeUB*capacity)+1))
    smallItemTypes = list(range(int(smallLB*capacity)+1,int(smallUB*capacity)+1))

    nBins = nItems
    nLargeItem = int(nItems*largeRatio)
    largeItems = np.random.choice(largeItemTypes,size=nLargeItem,replace=False)
    smallItems = np.random.choice(smallItemTypes,size=nItems - nLargeItem,replace=False)
    items = np.concatenate([largeItems,smallItems],axis=0)
    np.random.shuffle(items)

    m = scip.Model()
    # set variables
    x = np.zeros((nItems,nBins)).astype(object)
    y = np.zeros(nBins).astype(object)

    for j in range(nBins):
        y[j] = m.addVar(f'y_{j}','B')
        for i in range(nItems):
            x[i,j] = m.addVar(f'x_{i}_{j}','B')

    # add constraints
    for i in range(nItems):
        m.addCons( x[i,:].sum() == 1 )

    for j in range(nBins):
        m.addCons( (x[:,j]*items).sum() <= capacity * y[j] )


    # set objective

    m.setObjective(y.sum())

    return m


if __name__ == '__main__':

    insDir = './data/BPP/train/instances'
    nIns = 500
    os.makedirs(insDir,exist_ok=True)
    for i in range(nIns):

        m = gen_bin_packing_ins(capacity=100,largeRatio=0.3,largeLB=0.8,largeUB=1.0,smallLB=0.1,smallUB=0.3,nItems=20)

        m.writeProblem(os.path.join(insDir,f'bin_packing_{i}.lp'))


print('done')