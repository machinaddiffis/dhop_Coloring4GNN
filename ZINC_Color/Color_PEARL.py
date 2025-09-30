import argparse
import torch
from types import SimpleNamespace
from mlp import MLP
from model import construct_model
from torch import nn, optim
import pickle
from dataloader import DataLoader

import os.path as osp
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing
from coloring import color_count

torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser()
# runtime / io

parser.add_argument("--out_dir", type=str, default="pretrained/")


# data
parser.add_argument("--use_subset", type=int, default=1)
parser.add_argument("--train_bs", type=int, default=128)
parser.add_argument("--val_bs", type=int, default=128)
# train
parser.add_argument("--epochs", type=int, default=1600)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=3e-6)
parser.add_argument("--n_hops", type=int, default=6)
parser.add_argument("--warmup_steps", type=int, default=500)
parser.add_argument("--save_best", type=int, default=1)
parser.add_argument("--eval_fixed_noise", type=int, default=1,
                    help="Use fixed RNG for evaluation when basis=0 (deterministic eval)")
# model flags commonly used by your code
parser.add_argument("--residual", type=int, default=1)
parser.add_argument("--batch_norm", type=int, default=1)#for Pearl
parser.add_argument("--graph_norm", type=int, default=0)
parser.add_argument("--basis", type=int, default=0)
parser.add_argument("--num_samples", type=int, default=128)
# MLP hyperparams (keep defaults sensible)
parser.add_argument("--n_mlp_layers", type=int, default=3) #2 types of MLP
parser.add_argument("--mlp_hidden_dims", type=int, default=105)
parser.add_argument("--mlp_use_bn", type=int, default=1)
parser.add_argument("--mlp_use_ln", type=int, default=0)
parser.add_argument("--mlp_dropout", type=float, default=0.0)
parser.add_argument("--mlp_act", type=str, default="relu")
parser.add_argument("--pearl_act", type=str, default="relu")

#add
parser.add_argument("--target_dim", type=int, default=1)
parser.add_argument("--base_model", type=str, default="gine")
parser.add_argument("--n_base_layers", type=int, default=4)
parser.add_argument("--n_edge_types", type=int, default=3)
parser.add_argument("--node_emb_dims", type=int, default= 128)
parser.add_argument("--base_hidden_dims", type=int, default=128)
parser.add_argument("--pooling", type=str, default="add")
parser.add_argument("--gine_model_bn", type=bool, default=False) #for GINE
parser.add_argument("--pe_dims", type=int, default=37)
parser.add_argument("--sample_aggr_model_name", type=str, default="gin")
parser.add_argument("--n_sample_aggr_layers", type=int, default=8)#aggr layer for pearl-gnn
parser.add_argument("--pearl_mlp_out", type=int, default=37)
parser.add_argument("--sample_aggr_hidden_dims", type=int, default=40)
parser.add_argument("--pearl_k", type=int, default=1)#k hop aggregatate
parser.add_argument("--pearl_mlp_nlayers", type=int, default=1)
parser.add_argument("--pearl_mlp_hid", type=int, default=37)
parser.add_argument("--n_node_types", type=int, default=28)
parser.add_argument("--pe_aggregate", type=str, default="add")


parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
parser.add_argument("--seed", type=int, default=3)
args = parser.parse_args()

name=f"Color_PEARL_seed{args.seed}"

def PE_matrix(A: torch.Tensor, base: int = 1000) -> torch.Tensor:
    """
    A: torch.Tensor, shape (nRow, featureD), 每个元素是 [1, 40] 的整数
    base: 缩放因子，默认 1000
    返回: torch.FloatTensor, same shape as A
    """
    if not torch.is_tensor(A):
        raise TypeError("A must be a torch.Tensor")
    if A.dim() != 2:
        raise ValueError(f"A must be 2D, got shape {tuple(A.shape)}")

    nRow, featureD = A.shape
    device = A.device

    # 维度指数: 2 * i / D, i=1..D
    i = torch.arange(1, featureD + 1, device=device, dtype=torch.float32)   # [D]
    exp = 2.0 * i / float(featureD)                                         # [D]

    # div_term = base ** exp, 然后扩到 [nRow, D]
    base_f = torch.tensor(float(base), device=device)
    div_term = torch.pow(base_f, exp).unsqueeze(0).expand(nRow, -1)         # [nRow, D]

    A_f = A.to(torch.float32)
    even_mask = (A % 2 == 0)
    # 向量化分支：偶数 -> sin，奇数 -> cos
    PEs = torch.where(even_mask, torch.sin(A_f / div_term), torch.cos(A_f / div_term))
    return PEs

def lr_lambda_fn(total_steps: int, warmup: int):
    def _fn(curr: int) -> float:
        if curr < warmup:
            return curr / max(1, warmup)
        return max(0.0, (total_steps - curr) / max(1, total_steps - warmup))
    return _fn
def make_cfg(args: argparse.Namespace) -> SimpleNamespace:
    """A minimal cfg shim that matches keys used by your model/trainer code."""
    return SimpleNamespace(
        # runtime
        n_epochs=args.epochs,
        train_batch_size=args.train_bs,
        val_batch_size=args.val_bs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_warmup_steps=args.warmup_steps,
        residual=bool(args.residual),
        batch_norm=bool(args.batch_norm),
        graph_norm=bool(args.graph_norm),
        basis=bool(args.basis),
        num_samples=args.num_samples,
        # mlp
        n_mlp_layers=args.n_mlp_layers,
        mlp_hidden_dims=args.mlp_hidden_dims,
        mlp_use_bn=bool(args.mlp_use_bn),
        mlp_use_ln=bool(args.mlp_use_ln),
        mlp_dropout_prob=args.mlp_dropout,
        mlp_activation=args.mlp_act,
        pearl_act=args.pearl_act,
        # dataset
        use_subset=bool(args.use_subset),
        pe_method="pearl",  # keep simple; we only add Laplacian via transform
        target_dim=args.target_dim,
        base_model=args.base_model,
        n_base_layers=args.n_base_layers,
        n_edge_types=args.n_edge_types,
        node_emb_dims=args.node_emb_dims,
        base_hidden_dims=args.base_hidden_dims,
        pooling=args.pooling,
        gine_model_bn=args.gine_model_bn,
        pe_dims=args.pe_dims,
        sample_aggr_model_name=args.sample_aggr_model_name,
        n_sample_aggr_layers=args.n_sample_aggr_layers,
        pearl_mlp_out=args.pearl_mlp_out,
        sample_aggr_hidden_dims=args.sample_aggr_hidden_dims,
        pearl_k=args.pearl_k,
        pearl_mlp_nlayers=args.pearl_mlp_nlayers,
        pearl_mlp_hid=args.pearl_mlp_hid,
        n_node_types=args.n_node_types,
        pe_aggregate=args.pe_aggregate
    )

def create_mlp(cfg: SimpleNamespace, in_dims: int, out_dims: int) -> MLP:
    return MLP(
        cfg.n_mlp_layers, in_dims, cfg.mlp_hidden_dims, out_dims,
        cfg.mlp_use_bn, cfg.mlp_activation, cfg.mlp_dropout_prob
    )
def create_mlp_ln(cfg: SimpleNamespace, in_dims: int, out_dims: int, use_bias: bool = True) -> MLP:
    return MLP(
        cfg.n_mlp_layers, in_dims, cfg.mlp_hidden_dims, out_dims,
        cfg.mlp_use_ln, cfg.pearl_act, cfg.mlp_dropout_prob,
        norm_type="layer", NEW_BATCH_NORM=True, use_bias=use_bias
    )

def get_device(gpu: int) -> torch.device:
    if gpu is not None and gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
def run_epoch(model: nn.Module, loader, device: torch.device, optimizer: optim.Optimizer,
              scheduler: LambdaLR, criterion: nn.Module, basis: bool, num_samples: int, train: bool,
              eval_fixed_noise: bool, seed: int) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    n_items = 0

    g = None
    if not train and eval_fixed_noise:
        g = torch.Generator(device=device).manual_seed(seed)


    for batch in loader:


        batch = batch.to(device)

        # coloring
        color_perm=[]
        num_classes = batch.color.max().item() + 1
        for _ in range(num_samples):
            perm = torch.randperm(num_classes, device=device)
            x_permuted = perm[batch.color].squeeze(1)
            x_permuted=x_permuted/torch.max(x_permuted)
            color_perm.append(x_permuted)
        color_perm=torch.stack(color_perm, dim=0).to(device).t()
        color_PE=color_perm-0.5

        W_list = []
        s=0
        for i in range(len(batch.Lap)):#every laplacian matrix in  batch
            N = batch.Lap[i].shape[0]
            e=s+N
            W_list.append(color_PE[s:e])
            s=e

        if train:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch, W_list)  # shape [B]
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * batch.y.size(0)
        else:
            with torch.no_grad():
                pred = model(batch, W_list)
                loss = torch.nn.functional.l1_loss(pred, batch.y, reduction="sum")
                total += loss.item()
        n_items += batch.y.size(0)

    return total / max(1, n_items)

with open("./zinc_dataset/train.pkl", "rb") as f:
    train_set = pickle.load(f)


with open("./zinc_dataset/test.pkl", "rb") as f:
    test_set = pickle.load(f)


with open("./zinc_dataset/val.pkl", "rb") as f:
    val_set = pickle.load(f)

color_num=max(color_count(train_set,args.n_hops),max(color_count(test_set,args.n_hops),color_count(val_set,args.n_hops)))

train_loader = DataLoader(train_set, batch_size=args.train_bs, shuffle=True,
                                 num_workers=1, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=args.val_bs, shuffle=False,
                            num_workers=1, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_set, batch_size=args.val_bs, shuffle=False,
                                num_workers=1, pin_memory=True, persistent_workers=True)



device = get_device(args.gpu)


cfg = make_cfg(args)

###################################
kwargs = {
    "deg": None,
    "device": str(device),
    "residual": cfg.residual,
    "bn": cfg.batch_norm,
    "sn": cfg.graph_norm,
    "feature_type": "discrete",
}
def _mlp(in_d: int, out_d: int) -> MLP: #for base_model
    return create_mlp(cfg, in_d, out_d)

def _mlp_ln(in_d: int, out_d: int, use_bias: bool = True) -> MLP: #for pe model
    return create_mlp_ln(cfg, in_d, out_d, use_bias)
model = construct_model(cfg, (_mlp, _mlp_ln), **kwargs)

model.to(device)

criterion = nn.L1Loss(reduction="mean")
optimizer = optim.Adam(model.parameters(), 1e-3)
total_steps = len(train_loader) * args.epochs
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fn(total_steps, args.warmup_steps))


best_val = float("inf")
best_test = float("inf")
history = []



for epoch in range(1, args.epochs + 1):
    train_mae = run_epoch(model, train_loader, device, optimizer, scheduler, criterion,
                            basis=bool(args.basis), num_samples=args.num_samples,
                            train=True, eval_fixed_noise=bool(args.eval_fixed_noise), seed=args.seed)
    
    val_mae = run_epoch(model, val_loader, device, optimizer, scheduler, criterion,
                        basis=bool(args.basis), num_samples=args.num_samples,
                        train=False, eval_fixed_noise=bool(args.eval_fixed_noise), seed=args.seed)
    test_mae = run_epoch(model, test_loader, device, optimizer, scheduler, criterion,
                            basis=bool(args.basis), num_samples=args.num_samples,
                            train=False, eval_fixed_noise=bool(args.eval_fixed_noise), seed=args.seed)

    # current LR (from first group)
    lr_now = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "test_mae": float(test_mae),
        "lr": float(lr_now),
    })

    if val_mae < best_val:
        best_val = val_mae
        best_test = test_mae
        if args.save_best:
            torch.save(model.state_dict(), osp.join(args.out_dir, "best_model.pth"))

    print(f"Epoch {epoch:04d} | train {train_mae:.4f} | val {val_mae:.4f} | test {test_mae:.4f} | lr {lr_now:.3e}")

    # --- Save metrics to PKL
    log_path = osp.join(args.out_dir, f"{name}_train_log.pkl")
    summary = {"best_val": float(best_val), "best_test": float(best_test), "history": history, "args": vars(args)}
    with open(log_path, "wb") as f:
        pickle.dump(summary, f)

    print(f"Saved log to {log_path}")
    if args.save_best:
        print(f"Best model (by val) saved to {osp.join(args.out_dir, 'best_model.pth')}")