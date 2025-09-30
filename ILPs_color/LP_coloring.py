
from collections import deque
import numpy as np
import torch


def global_khop_coloring(edge_index: torch.Tensor, num_v: int, num_c: int, k: int):

    ei = edge_index.to(torch.long)

    r0max, r1max = int(ei[0].max().item()), int(ei[1].max().item())
    if r0max < num_v and r1max < num_c:
        L, R = ei[0], ei[1]
    elif r1max < num_v and r0max < num_c:
        L, R = ei[1], ei[0]
    else:
        raise ValueError("edge_index 的两行范围跟 (num_v, num_c) 对不上，检查输入。")

    N = num_v + num_c

    adj = [set() for _ in range(N)]
    for l, r in zip(L.tolist(), R.tolist()):
        u, v = int(l), num_v + int(r)
        if 0 <= u < num_v and num_v <= v < N:
            adj[u].add(v)
            adj[v].add(u)

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


    order = sorted(range(N), key=lambda x: len(conflict[x]), reverse=True)

    color = [-1] * N
    for u in order:
        used = {color[v] for v in conflict[u] if color[v] != -1}
        c = 0
        while c in used:
            c += 1
        color[u] = c


    color = torch.tensor(color, dtype=torch.long)
    col_L = color[:num_v]
    col_R = color[num_v:]
    return col_L, col_R

def PE_matrix(A, base=1000):

    nRow, featureD = A.shape

    dimInd = np.arange(1, featureD + 1)
    dimM = np.tile(dimInd, (nRow, 1))


    div_term = base ** (2 * dimM / featureD)

    even_mask = (A % 2 == 0)
    odd_mask = ~even_mask


    PEs = np.zeros_like(A, dtype=np.float32)

    PEs[even_mask] = np.sin(A[even_mask] / div_term[even_mask])
    PEs[odd_mask]  = np.cos(A[odd_mask]  / div_term[odd_mask])

    return PEs

if __name__ == '__main__':
    k=2
    d=32
    #toy graph
    num_v, num_c = 4, 3
    edge_index = torch.tensor([
        [0, 0, 1, 2, 3],
        [0, 1, 1, 2, 0]
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


