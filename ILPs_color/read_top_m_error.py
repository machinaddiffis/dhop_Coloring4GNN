import os
import numpy as np
import re
import scipy.io as io
import pandas

rootDir = r'./exp'
sampleTimes = 8
epoch=100
#for BPP
augs = ['empty','uniform','pos','orbit','group',"color","colorGNN","colorOrbit","colorGroup","empty32"]
datasets = ['BPP']

#for all
# augs = ['empty','uniform','pos',"color","orbit","group"]
#datasets = ['BPP','BIP','SMSP']


opt='opt'

expList = os.listdir(rootDir)

handisTable = pandas.DataFrame()
lossTable = pandas.DataFrame()
handisData = []
lossData = []

for dataset in datasets:
    for aug in augs:
        expInfo = [dataset,aug]

        expName = fr'dataset-{dataset}-Aug-{aug}-opt-{opt}-epoch-{epoch}-sampleTimes-{sampleTimes}'
        filepath = os.path.join(rootDir,expName,'loss_record.mat')
        data = io.loadmat(filepath)

        valid_loss = data['valid_loss'][0]
        # print(data.keys())
        print("-------")


        bestInd = valid_loss.argmin()

        x=data["train_loss"][0][bestInd]
        print(f"{aug},training_loss:{data['train_loss'][0][bestInd]},testing_loss:{data['valid_loss'][0][bestInd]},Generalization Gap:{-data['train_loss'][0][bestInd]+data['valid_loss'][0][bestInd]}")

        valid_handis = list(data['valid_handis'][bestInd])

        handisData.append(expInfo + valid_handis)





handisTable = pandas.DataFrame(handisData, columns=["Dataset", "Method", "Top-10%", "Top-20%", "Top-30%", "Top-40%", "Top-50%", "Top-60%", "Top-70%", "Top-80%", "Top-90%", "Top-100%"])


handisTable.to_excel('handisTable_valid.xlsx')

print('done')
