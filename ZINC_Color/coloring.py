import pickle

import os
from collections import deque
from torch_geometric.utils import to_undirected
import torch

def k_hop_random_coloring(g, k: int, seed: int = 0) -> torch.Tensor:
    """
    k-hop 随机染色：任意距离 <= k 的两个节点颜色不同。
    参数：
        g   : torch_geometric.data.Data，要求含 edge_index
        k   : int，k-hop
        seed: 可选随机种子，仅影响染色顺序
    返回：
        colors: [num_nodes, 1] 的 LongTensor，颜色从 0 开始
    """
    assert k >= 1, "k 至少为 1，不然叫染色只是涂心安。"
    num_nodes = g.num_nodes
    edge_index = to_undirected(g.edge_index, num_nodes=num_nodes)

    # 构建邻接表（无自环）
    neighbors = [set() for _ in range(num_nodes)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            neighbors[u].add(v)

    # 预计算每个节点的 <=k-hop 邻域（不含自身）
    k_neighbors = [set() for _ in range(num_nodes)]
    for s in range(num_nodes):
        visited = {s}
        q = deque([(s, 0)])
        while q:
            u, d = q.popleft()
            if d == k:
                continue
            for v in neighbors[u]:
                if v not in visited:
                    visited.add(v)
                    k_neighbors[s].add(v)
                    q.append((v, d + 1))

    # 随机顺序 + 贪心着色（选最小可用颜色）
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    order = torch.randperm(num_nodes, generator=gen).tolist()

    colors = [-1] * num_nodes
    for u in order:
        forbidden = {colors[v] for v in k_neighbors[u] if colors[v] != -1}
        c = 0
        while c in forbidden:
            c += 1
        colors[u] = c
    g.color=torch.tensor(colors, dtype=torch.long).view(-1, 1)
    return torch.tensor(colors, dtype=torch.long).view(-1, 1)


def color_count(dataset,hop=2):
    max_color = 0
    for iter_data in dataset:
        color=k_hop_random_coloring(iter_data,hop)
        term_color=torch.unique(color).numel()
        if term_color>max_color:
            max_color=term_color
    return max_color

# 可选：简单校验函数，出事儿了别怪我没给保险




