# Feature Augmentation of GNNs for ILPs: Local Uniqueness Suffices


We provide implementations and scripts for three tasks/datasets:

1. **ILPs**, 
    Each uses a slightly different environment and workflow, detailed below.
## Requirements

To avoid dependency conflicts, create **two separate environments**: one for ILPs and one for ZINC.

| Scenario        | Python  | PyTorch | PyG   | Other               | CUDA            |
|-----------------|---------|---------|-------|---------------------|-----------------|
| **ILPs **  | 3.8.13  | 1.10.2  | 2.0.4 | `pyscipopt 4.2.0`   | `cudatoolkit 11.3` |



## Installation Tips
Below are illustrative commands. Choose the correct wheels/indexes for your OS/CUDA combination.

```bash

# ILPs environment
conda create -n ilp python=3.8.13 -y
conda activate ilp
# Install specific versions of torch / pyg as required for your platform
# pip install torch==1.10.2+cu113 ...
# pip install torch-geometric==2.0.4
pip install pyscipopt==4.2.0
```

---

> Make sure the CUDA Toolkit version matches your installed PyTorch build.
## Data preparation

**ILPs**: instructions [here](./DATA.md) to prepare the ILPs data.


## Training the Model

To train the model , you can use the following bash commands:

**ILPs**：

```
epoch=100  
sampleTimes=8  
for dataset in BIP BPP SMSP  
do  
    python train.py --Aug empty --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes  
    python train.py --Aug uniform --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes  
    python train.py --Aug pos --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes  
    python train.py --Aug orbit --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes  
    python train.py --Aug group --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes  
    python train.py --Aug color --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes  
done  
​  
for dataset in BPP  
do  
    python train.py --Aug empty32 --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes  
    python train.py --Aug colorOrbit --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes  
    python train.py --Aug colorGroup --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes  
    python train.py --Aug colorGNN --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes  
done




## Evaluation

### ILP:Get Top-m% error

statistics regarding Top-m% error can be calculated by running

python read_top_m_error.py

the results will be reported in `./handisTable_valid.xlsx`



