
from collections import deque
import numpy as np
import torch


def global_khop_coloring(edge_index: torch.Tensor, num_v: int, num_c: int, k: int):
    """
    在合并图上做 k-hop 染色：任意两点若最短路距离 <= k，则不得同色。
    edge_index: shape [2, E]，两行分别是某侧的索引；自动判别哪行是左哪行是右。
    返回:
      col_L: [num_v]  左侧颜色
      col_R: [num_c]  右侧颜色（独立编号但与左侧共用同一调色盘）
    """
    ei = edge_index.to(torch.long)

    # 自动判定哪一行是 L 哪一行是 R（用最大值范围判定）
    r0max, r1max = int(ei[0].max().item()), int(ei[1].max().item())
    if r0max < num_v and r1max < num_c:
        L, R = ei[0], ei[1]
    elif r1max < num_v and r0max < num_c:
        L, R = ei[1], ei[0]
    else:
        raise ValueError("edge_index 的两行范围跟 (num_v, num_c) 对不上，检查输入。")

    N = num_v + num_c
    # 建邻接表（合并图，右侧偏移）
    adj = [set() for _ in range(N)]
    for l, r in zip(L.tolist(), R.tolist()):
        u, v = int(l), num_v + int(r)
        if 0 <= u < num_v and num_v <= v < N:
            adj[u].add(v)
            adj[v].add(u)

    # 预计算每个点的 k-hop 冲突集合（不含自身）
    conflict = [set() for _ in range(N)]
    for s in range(N):
        q = deque([(s, 0)])
        visited = {s}
        while q:
            u, d = q.popleft()
            if d == k:
                continue
            for w in adj[u]:
                if w not in visited:
                    visited.add(w)
                    q.append((w, d + 1))
        visited.remove(s)
        conflict[s] = visited

    # 按冲突度从大到小贪心染色
    order = sorted(range(N), key=lambda x: len(conflict[x]), reverse=True)
    # import random
    # nodes = list(range(N))
    # random.shuffle(nodes)  # 随机打乱一遍
    # order = sorted(nodes, key=lambda x: len(conflict[x]), reverse=True)
    color = [-1] * N
    for u in order:
        used = {color[v] for v in conflict[u] if color[v] != -1}
        c = 0
        while c in used:
            c += 1
        color[u] = c
    # used = {color[v] for v in conflict[u] if color[v] != -1}
    # candidates = [c for c in range(N) if c not in used]
    # color[u] = random.choice(candidates)

    color = torch.tensor(color, dtype=torch.long)
    col_L = color[:num_v]
    col_R = color[num_v:]
    return col_L, col_R

def PE_matrix(A, base=1000):
    """
    A: numpy array, shape (420, 32), 每个元素是 [1, 40] 的整数
    base: 缩放因子，默认 1000，保持和 PFs 一致
    """
    nRow, featureD = A.shape
    # 构造维度索引矩阵 dimM (nRow, featureD)
    dimInd = np.arange(1, featureD + 1)
    dimM = np.tile(dimInd, (nRow, 1))

    # 按 PFs 的公式计算分母 (指数缩放)
    div_term = base ** (2 * dimM / featureD)

    # 奇偶 mask
    even_mask = (A % 2 == 0)
    odd_mask = ~even_mask

    # 初始化结果矩阵
    PEs = np.zeros_like(A, dtype=np.float32)

    # 填充 sin/cos
    PEs[even_mask] = np.sin(A[even_mask] / div_term[even_mask])
    PEs[odd_mask]  = np.cos(A[odd_mask]  / div_term[odd_mask])

    return PEs

if __name__ == '__main__':
    k=2#k染色层
    d=32#染色维数
    #toy graph
    num_v, num_c = 4, 3 #variable的数量，constraints的数量
    edge_index = torch.tensor([
        [0, 0, 1, 2, 3],  # variable节点 id
        [0, 1, 1, 2, 0]  # constraints节点 id
    ])


    col_L, col_R = global_khop_coloring(edge_index, num_v, num_c, k=k)
    color_v = []
    color_c = []
    ccc = torch.cat([col_L, col_R])
    num_classes = ccc.unique().numel()
    for _ in range(d):
        v_permuted = col_L
        c_permuted = col_R
        color_v.append(v_permuted)
        color_c.append(c_permuted)
    color_vf = torch.stack(color_v, dim=0).t()
    color_cf = torch.stack(color_c, dim=0).t()

    color_cf = PE_matrix(color_cf)#concat to variable nodes input
    color_vf = PE_matrix(color_vf)#concat to constraints nodes input


