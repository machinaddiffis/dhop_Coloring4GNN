import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
from dataset import MIPDataset, SeedGenerator

import gzip
import pickle

from config import *
import argparse
from coloring import global_khop_coloring
from feature_aug import PE_matrix

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

torch.manual_seed(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def process_one(item, info, bgdir, k=3):
    inspath, solpath = item
    try:
        insname = os.path.basename(inspath)
        bpSaveDir = os.path.join(info['trainDir'], f'bipartites_PreColor')
        os.makedirs(bpSaveDir, exist_ok=True)
        filename_new = os.path.join(bpSaveDir, f'{insname}.bp')

        if os.path.exists(filename_new):
            return f"file exist：{filename_new}"

        bgpath = os.path.join(bgdir, insname + '.bp')

        data = pickle.load(gzip.open(bgpath, 'rb'))

        #Both are OK. if get error choose anthor 1
        #NAME 1
        # vf = torch.Tensor(data['variableFeatures'])
        # cf = torch.Tensor(data['constraintFeatures'])
        # edge = torch.Tensor(data["edgeInds"])
        #NAME 2
        vf = data['varFeatures']
        cf = data['consFeatures']
        edge = data["edgeInds"]
        v_num = vf.shape[0]
        c_num = cf.shape[0]
        col_L, col_R = global_khop_coloring(edge, v_num, c_num, k=k)

        color_v = []
        color_c = []
        ccc = torch.cat([col_L, col_R])
        num_classes = ccc.unique().numel()


        for _ in range(32):
            perm = torch.randperm(num_classes)
            v_permuted = col_L
            c_permuted = col_R
            _ = torch.max(torch.max(v_permuted), torch.max(c_permuted))
            color_v.append(v_permuted)
            color_c.append(c_permuted)

        color_vf = torch.stack(color_v, dim=0).t()
        color_cf = torch.stack(color_c, dim=0).t()

        color_cf = PE_matrix(color_cf)
        color_vf = PE_matrix(color_vf)

        data['groupFeatures'] = torch.Tensor(color_vf)
        data['consColorFeatures'] = torch.Tensor(color_cf)

        with gzip.open(filename_new, "wb") as f:
            pickle.dump(data, f)

        return f"processed: {insname}"
    except Exception as e:

        return f"ERROR {os.path.basename(inspath)}: {repr(e)}"

if __name__ == '__main__':

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--expName', type=str, default='exp')
    parser.add_argument('--dataset', type=str, default='SMSP')
    parser.add_argument('--opt', type=str, default='opt')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--Aug', type=str, default='color')
    parser.add_argument('--sampleTimes', type=int, default=8)
    parser.add_argument('--nproc', type=int, default=min(60, os.cpu_count() or 1),
                        help='default  min(60, cpu_count())')
    args = parser.parse_args()

    LEARNING_RATE = 0.0001
    SAMPLE_TIMES = args.sampleTimes
    NB_EPOCHS = args.epoch
    PRT_FREQUENCY = 1
    BATCH_SIZE = 1
    TBATCH = 1
    NUM_WORKERS = 1
    OPT = args.opt
    exp_dir = os.path.join(args.expName, f'dataset-{args.dataset}-Aug-{args.Aug}-opt-{OPT}-epoch-{NB_EPOCHS}-sampleTimes-{args.sampleTimes}')

    info = confInfo[args.dataset]
    DIR_INS = os.path.join(info['trainDir'], 'instances')
    DIR_SOL = os.path.join(info['trainDir'], 'solutions')
    DIR_BG = os.path.join(info['trainDir'], 'bipartites')
    NGROUP = info['nGroup']
    REORDER = info['reorder']
    augFunc = info['featureAugFuncs'][args.Aug]

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    os.makedirs(exp_dir, exist_ok=True)
    sample_names = os.listdir(DIR_SOL)
    sample_files = [(os.path.join(DIR_INS, name.replace('.sol', '')), os.path.join(DIR_SOL, name)) for name in sample_names]

    random.seed(0)
    random.shuffle(sample_files)

    train_files = sample_files


    trSeedGenerators = SeedGenerator(10)
    train_data = MIPDataset(train_files, DIR_BG, REORDER, augFunc, SAMPLE_TIMES, trSeedGenerators)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    bgdir = DIR_BG

    files = train_files

    k = 2

    def need_process(item):
        inspath, _ = item
        insname = os.path.basename(inspath)
        filename_new = os.path.join(info['trainDir'], f'bipartites_PreColor', f'{insname}.bp')
        return not os.path.exists(filename_new)

    todo = [it for it in files if need_process(it)]
    skipped = len(files) - len(todo)
    if skipped > 0:

        for it in files:
            inspath, _ = it
            insname = os.path.basename(inspath)
            filename_new = os.path.join(info['trainDir'], f'bipartites_PreColor', f'{insname}.bp')
            if os.path.exists(filename_new):
                print(f"file exist：{filename_new}")

    nproc = max(1, int(args.nproc))
    if nproc == 1:

        for i, it in enumerate(todo, 1):
            msg = process_one(it, info, bgdir, k=k)
            print(msg)
    else:
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futs = [ex.submit(process_one, it, info, bgdir, k) for it in todo]
            done = 0
            total = len(futs)
            for fut in as_completed(futs):
                done += 1
                msg = fut.result()
                print(msg)

                if done % 20 == 0 or done == total:
                    print(f'processed {done}/{total} (+ skipped {skipped})')