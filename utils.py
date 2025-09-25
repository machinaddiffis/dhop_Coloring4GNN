import numpy as np
import torch
import re
import math
import random






def reorderBP(names):
    nItem = max([int(re.findall('\d+', name)[0]) for name in names if 'x' in name]) + 1
    nBin = max([int(re.findall('\d+', name)[1]) for name in names if 'x' in name]) + 1

    XOrder = torch.Tensor(nItem+1 , nBin)

    for ind, name in enumerate(names):
        ss = re.findall('\d+', name)
        if 'x' in name:
            a, b = int(ss[0]), int(ss[1])
            XOrder[a, b] = ind
        if 'y' in name:
            b = int(ss[0])
            XOrder[-1, b] = ind


    return {
        'reorderInds': XOrder,
        'nGroup': nBin,
        'nElement': nItem+1
    }

def reorderSMSP(names):

    nItem = max([int(re.findall('\d+', name)[0]) for name in names if 'X' in name])+1
    nCap = max([int(re.findall('\d+', name)[0]) for name in names if 'Y' in name])+1

    XOrder = torch.Tensor(nItem+nCap,nItem)

    for ind,name in enumerate(names):
        ss = re.findall('\d+',name)
        a,b = int(ss[0]),int(ss[1])
        if 'X' in name:
            XOrder[a,b] = ind
        elif 'Y' in name:
            XOrder[a+nItem,b] = ind



    return {
        'reorderInds':XOrder,
        'nGroup':nItem,
        'nElement':nItem+nCap
    }


def reorderIP(names):
    nItem = max([int(re.findall('\d+', name)[0]) for name in names if 'place' in name]) + 1
    nBin = max([int(re.findall('\d+', name)[1]) for name in names if 'place' in name]) + 1

    XOrder = torch.Tensor(nItem , nBin)

    for ind, name in enumerate(names):
        if 'place' not in name:
            continue
        ss = re.findall('\d+', name)
        a, b = int(ss[0]), int(ss[1])

        XOrder[a, b] = ind

    return {
        'reorderInds': XOrder,
        'nGroup': nBin,
        'nElement': nItem
    }

def generatePosVector(pos1d,n,d):
    pos = np.zeros(n)
    i = int(pos1d*n*d)
    nPos = i//d
    dPos = i-nPos*d
    v = dPos/d + 1/d
    pos[nPos] = v
    return pos






if __name__ == '__main__':

    x = torch.Tensor([1,5,6,7,8,2,3,4,9])
    names = ['X_1_1','X_2_2','X_2_3','X_3_1','X_3_2','X_1_2','X_1_3','X_2_1','X_3_3']

    reorderExample(names)
    print('done')