import random

import numpy as np
import copy
from feature_extract import extract_features
import torch
from torch.utils.data import Dataset
import os
import gzip
import pickle
from config import *

import argparse

class SeedGenerator:
    def __init__(self,initialSeed = 0,nSeeds=1000000):

        random.seed(initialSeed)
        self.initialSeed = initialSeed
        self.fixedSeeds = [random.randint(1,100000000) for i in range(nSeeds)]
        self.cur_ind = 0
        print(f'set {len(self.fixedSeeds)} seeds, initialSeed {initialSeed}')
    def get_seed(self):
        self.cur_ind += 1
        return self.fixedSeeds[self.cur_ind-1]


class MIPDataset(Dataset):
    def __init__(self,files,bgdir,reorderFunc, augFunc,sampleTimes,seedGenerator):
        insPaths = [ filepaths[0] for filepaths in files]
        solPaths = [ filepaths[1] for filepaths in files]
        self.insPaths = insPaths
        self.solPaths = solPaths
        self.bgdir = bgdir
        self.reorder = reorderFunc
        self.addPos = augFunc
        random.seed(seedGenerator.initialSeed)
        self.seeds = [random.randint(1,5000) for _ in range(sampleTimes)] if sampleTimes>0 else None
        os.makedirs(bgdir,exist_ok=True)
        self.seedGenerator = seedGenerator
        # self.seedSets = set()

    def __getitem__(self, index):
        inspath = self.insPaths[index]
        solpath = self.solPaths[index]

        insname = os.path.basename(inspath)
        bgpath = os.path.join(self.bgdir,insname+'.bp')

        bpData = pickle.load(gzip.open(bgpath,'rb'))
        if 'sols' not in bpData.keys():
            solData = pickle.load(gzip.open(solpath, 'rb'))
            reorderData = self.reorder(bpData['varNames'])
            data = {
                # 'groupFeatures':torch.Tensor(features.groupFeatures),
                'varFeatures': torch.Tensor(bpData['variableFeatures']),
                'consFeatures': torch.Tensor(bpData['constraintFeatures']),
                'edgeFeatures': torch.Tensor(bpData['edgeWeights']),
                'edgeInds': torch.Tensor(bpData['edgeInds'].astype(int)).permute(1, 0), # var ID -> cons ID
                'nGroup': reorderData['nGroup'],
                'nElement': reorderData['nElement'],
                'reorderInds': torch.Tensor(reorderData['reorderInds'])
            }


            sols = solData['sols']
            objs = solData['objs']
            if ''.join(bpData['varNames']) != ''.join(solData['varNames']):
                raise NotImplementedError
            data['sols'] = torch.Tensor(sols[0])
            data['objs'] = torch.Tensor([objs[0]])


            pickle.dump(data,gzip.open(bgpath,'wb'))
            bpData = data

        # add aug features
        generated_seed = self.seedGenerator.get_seed()

        if self.seeds is None:
            data = self.addPos(bpData, seed=generated_seed)
        else:
            random.seed(generated_seed)
            selectedSeed = random.choice(self.seeds)
            selectedSeed = selectedSeed*(index+1)
            data = self.addPos(bpData, seed=selectedSeed)
            # self.seedSets.add(selectedSeed)

        return data

    def __len__(self):
        return len(self.insPaths)


class MIPDataset_SMSP(Dataset):
    def __init__(self, files, bgdir, reorderFunc, augFunc, sampleTimes, seedGenerator):
        insPaths = [filepaths[0] for filepaths in files]
        solPaths = [filepaths[1] for filepaths in files]
        self.insPaths = insPaths
        self.solPaths = solPaths
        self.bgdir = bgdir
        self.reorder = reorderFunc
        self.addPos = augFunc
        random.seed(seedGenerator.initialSeed)
        self.seeds = [random.randint(1, 5000) for _ in range(sampleTimes)] if sampleTimes > 0 else None
        os.makedirs(bgdir, exist_ok=True)
        self.seedGenerator = seedGenerator
        # self.seedSets = set()

    def __getitem__(self, index):
        inspath = self.insPaths[index]
        solpath = self.solPaths[index]

        insname = os.path.basename(inspath)
        bgpath = os.path.join(self.bgdir, insname + '.bp')

        bpData = pickle.load(gzip.open(bgpath, 'rb'))
        if 'sols' not in bpData.keys():
            solData = pickle.load(gzip.open(solpath, 'rb'))
            reorderData = self.reorder(bpData['varNames'])
            data = {
                # 'groupFeatures':torch.Tensor(features.groupFeatures),
                'varFeatures': torch.Tensor(bpData['variableFeatures']),
                'consFeatures': torch.Tensor(bpData['constraintFeatures']),
                'edgeFeatures': torch.Tensor(bpData['edgeWeights']),
                'groupFeatures': torch.Tensor(bpData['groupFeatures']),
                'consColorFeatures': torch.Tensor(bpData['consColorFeatures']),
                'edgeInds': torch.Tensor(bpData['edgeInds'].astype(int)).permute(1, 0),  # var ID -> cons ID
                'nGroup': reorderData['nGroup'],
                'nElement': reorderData['nElement'],
                'reorderInds': torch.Tensor(reorderData['reorderInds'])
            }
            sols = solData['sols']
            objs = solData['objs']
            if ''.join(bpData['varNames']) != ''.join(solData['varNames']):
                raise NotImplementedError
            data['sols'] = torch.Tensor(sols[0])
            data['objs'] = torch.Tensor([objs[0]])

            pickle.dump(data, gzip.open(bgpath, 'wb'))
            bpData = data

        return bpData

    def __len__(self):
        return len(self.insPaths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SMSP')
    args = parser.parse_args()
    info = confInfo[args.dataset]
    ADDPOS = info['addPosFeature']
    REORDER = info['reorder']
    fileDir = os.path.join(info['trainDir'], 'ins')
    solDir = os.path.join(info['trainDir'], 'sol')
    bgDir = os.path.join(info['trainDir'], 'bg')
    solnames = os.listdir(solDir)
    filepaths = [os.path.join(fileDir, solname.replace('.sol', '')) for solname in solnames]
    solpaths = [os.path.join(solDir, solname) for solname in solnames]
    dataset = MIPDataset(list(zip(filepaths,solpaths)),bgDir,REORDER,ADDPOS)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Start constructing bipartite graph ...')
    for step,data in enumerate(data_loader):
        varFeatures = data['varFeatures']
        consFeatures = data['consFeatures']
        edgeFeatures = data['edgeFeatures']
        edgeInds = data['edgeInds']
        sols = data['sols']
        objs = data['objs']
        reorderInds = data['reorderInds']

        print(f'Processed {step}/{len(data_loader)}')
    print('Bipartite graph construction finished!')