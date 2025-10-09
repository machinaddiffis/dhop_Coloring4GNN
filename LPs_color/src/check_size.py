from model import *
import torch 
import gzip
import pickle
import os
import random
import argparse
from alive_progress import alive_bar
import math
from LP_coloring import *

for k in ['pagerank','lb','ip']:
    ftrain = gzip.open(f'../data/{k}/train/packingdata0.pkl','r')
    p = pickle.load(ftrain)
    tr_pr = p['label'].shape
    tr_du = p['dual'].shape
    ftrain.close()
    ftrain = gzip.open(f'../data/{k}/test/packingdata0.pkl','r')
    p = pickle.load(ftrain)
    tt_pr = p['label'].shape
    tt_du = p['dual'].shape
    ftrain.close()
    print(f' {k}:   train: {tr_pr[0]} X {tr_du[0]}\n            test: {tt_pr[0]} X {tt_du[0]}')