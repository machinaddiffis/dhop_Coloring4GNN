import torch
import torch_geometric
import torch.nn as nn


EMB_SIZE=64


class GNNPolicy(torch.nn.Module):
    '''
        from Gasse's  model. see https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation/example.ipynb
    '''
    def __init__(self,nGroup):
        super().__init__()
        emb_size = EMB_SIZE
        cons_nfeats = 2
        edge_nfeats = 1
        var_nfeats = 7
        group_nfeats = nGroup
        # GROUP EMBEDDING
        self.group_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(group_nfeats),
            torch.nn.Linear(group_nfeats, emb_size),
            torch.nn.ReLU(),
        )

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()
        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features,group_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0) # cons ID -> var ID


        group_features = self.group_embedding(group_features)


        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features) + group_features

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output



class GNNPolicy32(torch.nn.Module):
    '''
        from Gasse's  model. see https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation/example.ipynb
    '''
    def __init__(self,nGroup):
        super().__init__()
        emb_size = EMB_SIZE
        cons_nfeats = 2
        edge_nfeats = 1
        var_nfeats = 7
        group_nfeats = nGroup
        # GROUP EMBEDDING
        self.group_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(group_nfeats),
            torch.nn.Linear(group_nfeats, emb_size),
            torch.nn.ReLU(),
        )

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size*32),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size*32, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()
        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features,group_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0) # cons ID -> var ID


        group_features = self.group_embedding(group_features)


        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features) + group_features

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output


class ColorGNNPolicy(torch.nn.Module):
    '''
        from Gasse's  model. see https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation/example.ipynb
    '''
    def __init__(self,nGroup):
        super().__init__()
        emb_size = EMB_SIZE
        cons_nfeats = 2
        edge_nfeats = 1
        var_nfeats = 7
        group_nfeats = nGroup

        # GROUP EMBEDDING
        self.group_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(group_nfeats),
            torch.nn.Linear(group_nfeats, emb_size),
            torch.nn.ReLU(),
        )
        self.consColor_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(group_nfeats),
            torch.nn.Linear(group_nfeats, emb_size),
            torch.nn.ReLU(),
        )

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )



        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()
        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features,group_features,consColor
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0) # cons ID -> var ID


        group_features = self.group_embedding(group_features)
        cons_color=self.consColor_embedding(consColor)


        # First step: linear embedding layers to a common dimension (64)

        constraint_features = self.cons_embedding(constraint_features)+cons_color
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features) + group_features

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output

class ColorNet(torch.nn.Module):
    '''
        from Gasse's  model. see https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation/example.ipynb
    '''
    def __init__(self,nGroup):
        super().__init__()
        emb_size = EMB_SIZE
        cons_nfeats = 2
        edge_nfeats = 1
        var_nfeats = 7
        group_nfeats = nGroup

        # GROUP EMBEDDING
        self.group_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(group_nfeats),
            torch.nn.Linear(group_nfeats, emb_size),
            torch.nn.ReLU(),
        )
        self.consColor_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(group_nfeats),
            torch.nn.Linear(group_nfeats, emb_size),
            torch.nn.ReLU(),
        )

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )



        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = ColorBipartiteGraphConvolution()
        self.conv_c_to_v = ColorBipartiteGraphConvolution()
        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.color_embedding=color_layer()
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )



    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features,variableColor,consColor,group_features=None
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0) # cons ID -> var ID


        if group_features is not None:

            group_features = self.group_embedding(group_features)


        # First step: linear embedding layers to a common dimension (64)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        if group_features is not None:
            variable_features = self.var_embedding(variable_features)+group_features
        else:
            variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features,color=consColor,output_module=self.color_embedding
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features,color=variableColor,output_module=self.color_embedding
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output




class ColorNet_emb(torch.nn.Module):
    '''
        from Gasse's  model. see https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation/example.ipynb
    '''
    def __init__(self,nGroup):
        super().__init__()
        emb_size = EMB_SIZE
        cons_nfeats = 2
        edge_nfeats = 1
        var_nfeats = 7
        group_nfeats = nGroup

        # GROUP EMBEDDING
        self.group_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(group_nfeats),
            torch.nn.Linear(group_nfeats, emb_size),
            torch.nn.ReLU(),
        )
        self.consColor_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(group_nfeats),
            torch.nn.Linear(group_nfeats, emb_size),
            torch.nn.ReLU(),
        )
        self.group_embedding_list=color_em(var_nfeats=var_nfeats)
        self.consColor_embedding_list=color_em(var_nfeats=cons_nfeats)
        self.PE_embedding_list = color_em(var_nfeats=32)



        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )




        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()
        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.color_embedding=color_layer()
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )



    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features,variableColor,consColor,group_features=None
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0) # cons ID -> var ID

        # perm = torch.randperm(32, device=constraint_features.device)
        # variableColor=perm[variableColor]
        # consColor=perm[consColor]

        if group_features is not None:
            group_features = self.group_embedding(group_features)
            # color_group = group_value_indices(variableColor)
            # Z0 = torch.zeros([group_features.shape[0], EMB_SIZE], device=group_features.device)
            # for color, index in color_group:
            #     same_color_v = self.PE_embedding_list(group_features[index, :], color)
            #     Z0[index, :] = same_color_v.clone()
            #
            # group_features=Z0

        # First step: linear embedding layers to a common dimension (64)
        edge_features = self.edge_embedding(edge_features)

        color_group = group_value_indices(variableColor)
        Z1 = torch.zeros([variable_features.shape[0],EMB_SIZE], device=variable_features.device)
        for color, index in color_group:
            same_color_v = self.group_embedding_list(variable_features[index, :], color)
            Z1[index, :] = same_color_v.clone()

        if group_features is not None:
            variable_features = Z1+group_features
        else:
            variable_features = Z1

        #cons
        color_group = group_value_indices(consColor)
        Z2 = torch.zeros([constraint_features.shape[0], EMB_SIZE], device=constraint_features.device)
        for color, index in color_group:
            same_color_c = self.consColor_embedding_list(constraint_features[index, :], color)
            Z2[index, :] = same_color_c.clone()
        constraint_features =Z2



        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output

def make_output_module(emb_size):
    return torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )



class color_layer(nn.Module):
    def __init__(self, emb_size=EMB_SIZE, num_modules=32):
        super().__init__()
        self.mo_list = nn.ModuleList([make_output_module(emb_size) for _ in range(num_modules)])

    def forward(self, x, idx):
        # 这里 idx 是选择第几个子模块
        return self.mo_list[idx](x)

def make_embedding_module(emb_size,var_nfeats=32):
    return  torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
class color_em(nn.Module):
    def __init__(self, emb_size=EMB_SIZE,var_nfeats=7, num_modules=32):
        super().__init__()
        self.mo_list = nn.ModuleList([make_embedding_module(emb_size,var_nfeats) for _ in range(num_modules)])

    def forward(self, x, idx):
        # 这里 idx 是选择第几个子模块
        return self.mo_list[idx](x)


class ColorBipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = EMB_SIZE

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features,color=None,output_module=None):
        """
        This method sends the messages, computed in the message method.
        if flow = source_to_target (default setting), edge_indices are [j,i], where i is the central node
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        concat_f=torch.cat([self.post_conv_module(output), right_features], dim=-1)


        color_group=group_value_indices(color)
        Z = torch.zeros(right_features.shape, device=right_features.device)
        for color,index in color_group:
            same_color_f=output_module(concat_f[index,:], color)
            Z[index,:]=same_color_f.clone()
        return Z

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = EMB_SIZE

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        if flow = source_to_target (default setting), edge_indices are [j,i], where i is the central node
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


def group_value_indices(x: torch.Tensor):
    """
    输入: 任意形状的整型/浮点型 torch.Tensor
    输出: [[value, [indices]], ...]
         其中 indices 是按扁平化后一维下标（0 基）。
         分组顺序按 value 在 x 中的首次出现顺序。
    """
    x = x.flatten()
    # 保留首次出现顺序
    vals, inv = torch.unique(x, sorted=False, return_inverse=True)
    groups = [[] for _ in range(len(vals))]
    for idx, gid in enumerate(inv.tolist()):
        groups[gid].append(idx)
    # 转成纯 Python 数字，避免后续序列化麻烦
    return [[vals[i].item() if vals[i].numel()==1 else vals[i].tolist(), groups[i]]
            for i in range(len(vals))]

