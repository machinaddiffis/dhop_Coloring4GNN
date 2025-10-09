# Feature Augmentation of GNNs for ILPs: Local Uniqueness Suffices


We provide implementations and scripts for three tasks/datasets:

1. **ILPs**, 2) **LPs**, and 3) **ZINC**.
  
    Each uses a slightly different environment and workflow, detailed below.
## Environment Setup
We recommend creating separate environments for the ILP/LP tasks and the ZINC task.
## Requirements

To avoid dependency conflicts, create **two separate environments**: one for ILPs/LPs and one for ZINC.

| Scenario        | Python  | PyTorch | PyG   | Other               | CUDA            |
|-----------------|---------|---------|-------|---------------------|-----------------|
| **ILPs / LPs**  | 3.8.13  | 1.10.2  | 2.0.4 | `pyscipopt 4.2.0`   | `cudatoolkit 11.3` |
| **ZINC**        | 3.11.13 | 2.7.1   | 2.6.1 | `ogb 1.3.6`         | `cudatoolkit 11.8` |


> Note: `ogb` refers to the Open Graph Benchmark Python package.

## Installation Tips
Below are illustrative commands. Choose the correct wheels/indexes for your OS/CUDA combination.

```bash

# ILPs / LPs environment
conda create -n ilp python=3.8.13 -y
conda activate ilp
# Install specific versions of torch / pyg as required for your platform
# pip install torch==1.10.2+cu113 ...
# pip install torch-geometric==2.0.4
pip install pyscipopt==4.2.0

  

# ZINC environment
conda create -n zinc python=3.11.13 -y

conda activate zinc

# pip install torch==2.7.1 ...

# pip install torch-geometric==2.6.1

pip install ogb==1.3.6
```

---

> Make sure the CUDA Toolkit version matches your installed PyTorch build.
## Data preparation

**ILPs**: instructions [here](./DATA.md) to prepare the ILPs data.

**LP**: - generate data by running:
```
python 1generate_pr.py
python 1generate_wa.py
```

**ZINC**: The dataset artifacts are created automatically when you launch the training script.

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
```

**LPs:**

```
#WA
python 2trainVanila.py --i color  -p lb
python 2trainVanila.py --i uniform  p lb
python 2trainVanila.py -i vanilla -p lb
#pagerank
python 2trainVanila.py --i color  -p pagerank
python 2trainVanila.py --i uniform  p pagerank
python 2trainVanila.py -i vanilla -p pagerank

```

**ZINC**:

```
python Color_PEARL.py
```

## Evaluation

### ILP:Get Top-m% error

statistics regarding Top-m% error can be calculated by running

python read_top_m_error.py

the results will be reported in `./handisTable_valid.xlsx`

### LP: Get MSE

python 3test.py

### ZINC:MAE 

Per-seed training logs are saved as:
```
Color_PEARL_seed{i}_train_log.pkl
```
where {i} is the seed index you used.

