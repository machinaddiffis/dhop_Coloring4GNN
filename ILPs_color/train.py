import os
os.environ['KMP_DUPLICATE_LIB_OK']="True"
import pyscipopt
from dataset import MIPDataset,SeedGenerator,MIPDataset_SMSP
from nn import GNNPolicy,ColorGNNPolicy,ColorNet,ColorNet_emb,GNNPolicy32

import scipy.io as io

from losses import labelOpt,lexOpt,get_han_loss

from config import *
import argparse

torch.manual_seed(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--expName', type=str, default='exp')
parser.add_argument('--dataset', type=str, default='BP')
parser.add_argument('--opt', type=str, default='opt')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--Aug', type=str, default='group')
parser.add_argument('--sampleTimes', type=int, default=-1)
args = parser.parse_args()

LEARNING_RATE = 0.0001
SAMPLE_TIMES = args.sampleTimes
NB_EPOCHS = args.epoch
PRT_FREQUENCY = 1
BATCH_SIZE = 1
TBATCH = 1
NUM_WORKERS = 0
OPT = args.opt
exp_dir = os.path.join(args.expName,f'dataset-{args.dataset}-Aug-{args.Aug}-opt-{OPT}-epoch-{NB_EPOCHS}-sampleTimes-{args.sampleTimes}')

info = confInfo[args.dataset]
DIR_INS = os.path.join(info['trainDir'],'instances')
DIR_SOL = os.path.join(info['trainDir'],'solutions')
DIR_BG = os.path.join(info['trainDir'],'bipartites')
if args.Aug=="color" and args.dataset=="SMSP":
    DIR_BG = os.path.join(info['trainDir'],f'bipartites_PreColor')
NGROUP = info['nGroup']
REORDER = info['reorder']
augFunc = info['featureAugFuncs'][args.Aug]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(exp_dir,exist_ok=True)
sample_names = os.listdir(DIR_SOL)
sample_files = [ (os.path.join(DIR_INS,name.replace('.sol','')),os.path.join(DIR_SOL,name)) for name in sample_names]



random.seed(0)
random.shuffle(sample_files)

train_files = sample_files[: int(0.6 * len(sample_files))]
valid_files = sample_files[int(0.6 * len(sample_files)) :]

trSeedGenerators = SeedGenerator(10)

##
if args.Aug=="color" and args.dataset=="SMSP":
    train_data = MIPDataset_SMSP(train_files,DIR_BG,REORDER,augFunc,SAMPLE_TIMES,trSeedGenerators)
else:
    train_data = MIPDataset(train_files,DIR_BG,REORDER,augFunc,SAMPLE_TIMES,trSeedGenerators)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

teSeedGenerators = SeedGenerator(20)
if args.Aug=="color" and args.dataset=="SMSP":
    valid_data = MIPDataset_SMSP(valid_files,DIR_BG,REORDER,augFunc,1,teSeedGenerators)
else:
    valid_data = MIPDataset(valid_files,DIR_BG,REORDER,augFunc,1,teSeedGenerators)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


if args.Aug == "empty32":  # Non-Aug+
    policy = GNNPolicy32(NGROUP).to(DEVICE)
elif args.Aug == "colorUID":
    policy = ColorGNNPolicy(NGROUP).to(DEVICE)
elif args.Aug in {"colorGNN", "colorOrbit", "colorGroup"}:
    policy = ColorNet_emb(NGROUP).to(DEVICE)
else:
    policy = GNNPolicy(NGROUP).to(DEVICE)


def process(policy, data_loader, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """

    if optimizer:
        policy.train()
    else:
        policy.eval()

    mean_loss = 999
    mean_acc = 0
    mean_han_diss = None

    n_samples_processed = 0
    pe_num_list=[]
    with torch.set_grad_enabled(optimizer is not None):
        batch_losses = []
        for step, batch in enumerate(data_loader):
            groupFeatures = batch['groupFeatures'][0].to(DEVICE)
            pe_num_list.append(groupFeatures[:, 0].unique().numel())
            print(groupFeatures[:, 0].unique().numel())
            varFeatures = batch['varFeatures'][0].to(DEVICE)
            consFeatures = batch['consFeatures'][0].to(DEVICE)
            edgeFeatures = batch['edgeFeatures'][0].to(DEVICE)
            edgeInds = batch['edgeInds'][0].to(DEVICE)
            sols = batch['sols'][0].to(DEVICE)
            objs = batch['objs'][0].to(DEVICE)
            reorderInds = batch['reorderInds'][0].long().reshape(-1)
            nGroup = batch['nGroup'][0]
            nElement = batch['nElement'][0]
            if args.Aug=="color":
                consColor=batch["consColorFeatures"][0].to(DEVICE)
                output = policy(
                    consFeatures,
                    edgeInds.long(),
                    edgeFeatures[:, None],
                    varFeatures,
                    groupFeatures,
                    consColor
                )
            else:
                if args.Aug=="colorNet":
                    consColor = batch["consColor"][0].to(DEVICE)
                    variableColor=batch["variableColor"][0].to(DEVICE)
                    output = policy(
                        consFeatures,
                        edgeInds.long(),
                        edgeFeatures[:, None],
                        varFeatures,
                        variableColor=variableColor,
                        consColor=consColor,
                    )
                else:
                    if args.Aug=="colorOrbit" or  args.Aug=="colorGroup":
                        consColor = batch["consColor"][0].to(DEVICE)
                        variableColor = batch["variableColor"][0].to(DEVICE)
                        output = policy(
                            consFeatures,
                            edgeInds.long(),
                            edgeFeatures[:, None],
                            varFeatures,
                            variableColor=variableColor,
                            consColor=consColor,
                            group_features=groupFeatures
                        )
                    else:

                        output = policy(
                            consFeatures,
                            edgeInds.long(),
                            edgeFeatures[:,None],
                            varFeatures,
                            groupFeatures

                        )
            output = output.sigmoid()


            X_hat = output[reorderInds].reshape(nElement,nGroup)
            X = sols[reorderInds].reshape(nElement,nGroup)
            #
            # # compute loss
            with torch.set_grad_enabled(True):
                opt_func = lexOpt if OPT=='lex' else labelOpt if OPT=='opt' else None
                X_bar = opt_func(X_hat.detach()[None,:,:], X.clone()[None,:,:],device=DEVICE)[0] if opt_func is not None else X


            sols[reorderInds] = X_bar.reshape(-1)

            pos_loss = -torch.log(output[reorderInds] + 0.00001) * (sols[reorderInds] >= 0.5)
            neg_loss = -torch.log(1 - output[reorderInds] + 0.00001) * (sols[reorderInds] < 0.5)
            loss = pos_loss.sum() + neg_loss.sum()

            if optimizer is not None:
                loss.backward()

            if step%TBATCH == TBATCH-1 or step==len(data_loader)-1:
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()
                # output
                # if step%PRT_FREQUENCY==0:
                #     mod = 'train' if optimizer else 'valid'
                #     print('Epoch {} {} [{}/{}] loss {:.6f}'.format( epoch, mod, step,len(data_loader),loss.item()))

            # hanming distance

            han_diss = get_han_loss(output[reorderInds].detach(),sols[reorderInds])



            mean_loss += loss.item() * X.shape[0]
            mean_han_diss = [hans + han_diss[ind] * X.shape[0] for ind, hans in enumerate(mean_han_diss)] if mean_han_diss is not None else han_diss
            #mean_acc += accuracy * batch.num_graphs
            n_samples_processed +=  X.shape[0]
    print("颜色数：",torch.sum(torch.tensor(pe_num_list))/len(pe_num_list))
    mean_loss /= n_samples_processed
    # mean_acc /= n_samples_processed
    mean_han_diss = [ hans/n_samples_processed for ind,hans in enumerate(mean_han_diss)]

    return mean_loss,mean_han_diss


optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []
tr_han_diss = []
val_han_diss = []

best_val_loss = 99999

for epoch in range(NB_EPOCHS):

    train_loss,tr_han_dis = process(policy, train_loader, optimizer)
    print(f"Epoch {epoch} Train loss: {train_loss:0.3f} han_dis: {tr_han_dis[-1]:.3f}")

    valid_loss,val_han_dis = process(policy, valid_loader, None)
    print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f} han_dis: {val_han_dis[-1]:.3f}")
    quit()

    if valid_loss<best_val_loss:
        best_val_loss = valid_loss

        torch.save(policy.state_dict(), os.path.join(exp_dir, 'model_best.pth'))
    torch.save(policy.state_dict(), os.path.join(exp_dir, 'model_last.pth'))

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    tr_han_diss.append(tr_han_dis)
    val_han_diss.append(val_han_dis)
io.savemat(os.path.join(exp_dir, 'loss_record.mat'), {
    'train_loss':np.array(train_losses),
    'valid_loss':np.array(valid_losses),
    'train_handis':np.array(tr_han_diss),
    'valid_handis':np.array(val_han_diss)
})


print('done')





