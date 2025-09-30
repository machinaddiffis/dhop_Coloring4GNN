from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing

from mlp import MLP

# This code is adapted from:
# https://github.com/Graph-COM/SPE/blob/master/src/gine.py 

class GINE(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int, out_dims: int,
        create_mlp: Callable[[int, int], MLP], bn: bool = False, residual: bool = False, feature_type: str = "discrete", pe_emb=37,
        bond_encoder=False
    ) -> None:
        """
        n_layers：模型中的图卷积层的数量。
        n_edge_types：边的类型数，这对处理不同类型的边特征非常重要。
        in_dims：输入节点特征的维度。
        hidden_dims：隐藏层的维度。
        out_dims：输出层的维度，通常是图或节点表示的维度。
        create_mlp：一个可调用的函数，用于创建多层感知机（MLP）。MLP 是用于节点表示转换的网络组件。
        bn：一个布尔值，指示是否使用批归一化（Batch Normalization）。
        residual：一个布尔值，表示是否使用残差连接。
        feature_type：表示特征的类型（例如“离散”特征）。
        pe_emb：位置编码的维度（通常用于表示节点的相对位置或图的拓扑信息）。
        bond_encoder：一个布尔值，表示是否使用额外的边编码信息（通常与分子结构相关）。
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.residual = residual
        self.bn = bn
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINELayer(n_edge_types, in_dims, hidden_dims, create_mlp, feature_type, pe_emb=pe_emb, bond_encoder=bond_encoder)
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))

        layer = GINELayer(n_edge_types, hidden_dims, out_dims, create_mlp, feature_type, pe_emb=pe_emb, bond_encoder=bond_encoder)
        self.layers.append(layer)

    def forward(self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, PE: torch.Tensor) -> torch.Tensor:
        """
        :param X_n: Node feature matrix. [N_sum, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param edge_attr: Edge type matrix. [E_sum]
        :return: Output node feature matrix. [N_sum, D_out]
        """

        for i, layer in enumerate(self.layers):
            X_0 = X_n
            X_n = layer(X_n, edge_index, edge_attr, PE)   # [N_sum, D_hid] or [N_sum, D_out]
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if X_n.ndim == 3:
                    X_n = self.batch_norms[i](X_n.transpose(2, 1)).transpose(2, 1)
                else:
                    X_n = self.batch_norms[i](X_n)
            if self.residual:
                X_n = X_n + X_0

        return X_n                                    # [N_sum, D_out]

# from ogb.graphproppred.mol_encoder import BondEncoder
class GINELayer(MessagePassing):
    edge_features: nn.Embedding
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, n_edge_types: int, in_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP],
                 feature_type: str = "discrete", pe_emb=37, bond_encoder=False) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        """
        n_edge_types：边的类型数量（例如，图中可能有不同类型的边，如单键、双键等）。
        in_dims：输入特征的维度（每个节点或边的特征维数）。
        out_dims：输出特征的维度（卷积后的节点表示维度）。
        create_mlp：一个函数，用来创建 MLP（多层感知机）。
        feature_type：边特征的类型，"discrete" 表示离散特征（使用 Embedding 层），"continuous" 表示连续特征（使用 Linear 层）。
        pe_emb：位置编码的维度（可以理解为图中节点的拓扑信息或位置）。
        bond_encoder：如果设置为 True，则使用 BondEncoder 来对边进行编码。
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)：调用父类 MessagePassing 的初始化函数，设置了图卷积操作中的聚合方法为加和（"add"），并指定了信息流的方向是从源节点到目标节点（"source_to_target"），node_dim=0 表示信息传递时，节点特征的维度为 0。
        """
        # super(GINELayer, self).__init__(aggr='add')
        # if bond_encoder:    
        #     self.edge_features = BondEncoder(emb_dim=in_dims)
        # else:
        #     self.edge_features = nn.Embedding(n_edge_types+1, in_dims) if feature_type == "discrete" else \
        #                     nn.Linear(n_edge_types, in_dims)# discrete的话会多出一个

      
        self.edge_features = nn.Embedding(n_edge_types+1, in_dims) if feature_type == "discrete" else \
                            nn.Linear(n_edge_types, in_dims)# discrete的话会多出一个

        # self.pe_embedding = nn.Linear(1, in_dims)
        self.pe_embedding = create_mlp(pe_emb, in_dims) # for pe-full
        # print(self.pe_embedding ,"pe")
        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True)
        self.mlp = create_mlp(in_dims, out_dims)

    def forward(self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                PE: torch.Tensor) -> torch.Tensor:
        """
        :param X_n: Node feature matrix. [N_sum, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param edge_attr: Edge type matrix. [E_sum]
        :return: Output node feature matrix. [N_sum, D_out]
        """

        X_e = self.edge_features(edge_attr) if edge_attr is not None else None # [E_sum, D_in] decide GINE or GIN?


        if PE is not None:
            if X_e is not None and PE.size(0) == X_e.size(0): # for PEG, num of edges == num of nodes
                X_e = X_e * self.pe_embedding(PE) if X_e is not None else self.pe_embedding(PE)
            else:

                X_n = X_n + self.pe_embedding(PE)


        # Contains sum(j in N(i)) {message(j -> i)} for each node i.
        S = self.propagate(edge_index, X=X_n, X_e=X_e)   # [N_sum, D_in]

        Z = (1 + self.eps) * X_n   # [N_sum, D_in]
        Z = Z + S                  # [N_sum, D_in]

        return self.mlp(Z)         # [N_sum, D_out]

    def message(self, X_j: torch.Tensor, X_e: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, D_in]
        :param X_e: Edge feature matrix. [E_sum, D_in]
        :return: The messages ReLU(X_j + E_ij) for each edge (j -> i). [E_sum, D_in]
        """
        return F.relu(X_j + X_e) if X_e is not None else F.relu(X_j)   # [E_sum, D_in]

