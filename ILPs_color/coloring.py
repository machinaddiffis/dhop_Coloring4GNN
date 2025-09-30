
from collections import deque

import torch

def global_khop_coloring(edge_index: torch.Tensor, num_v: int, num_c: int, k: int):
    """
    k-hop coloring：dist(u,v)<= k，different colors。
    edge_index: shape [2, E]，
    return:
      col_L: [num_v]  left color
      col_R: [num_c] rright color
    """
    ei = edge_index.to(torch.long)

    r0max, r1max = int(ei[0].max().item()), int(ei[1].max().item())
    if r0max < num_v and r1max < num_c:
        L, R = ei[0], ei[1]
    elif r1max < num_v and r0max < num_c:
        L, R = ei[1], ei[0]
    else:
        raise ValueError("edge_index error ,check input!。")

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

    # order!
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


