# Building GIN using the GIN layers

import torch
from torch.nn import (
    Linear,
    Sequential,
    BatchNorm1d,
    ReLU,
    Dropout,
    LeakyReLU,
)
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from graphnetwork.GIN_layers import GINE_layer, MLPembd

class GIN(torch.nn.Module):
    def __init__(
        self,
        in_channels_gin_x=4,
        in_channels_gin_edge=8,
        hidden_channels_gin=16,
        out_channels_gin=16,
        hidden_channels_global=2,
        out_channels_global=1,
        num_layers=1,
        edge_features=True,
        edge_dim=8,
        dropout=0.15,
        linear_learn=True,
        activation_function_gin="LeakyReLU",
        activation_function_mlp="LeakyReLU",
        aggregation_nodes_edges="max",
        aggregation_global="max",
        device="cuda:o",
    ):

        super().__init__()

        self.num_layers = num_layers
        self.gin_MLP_layers = torch.nn.ModuleList()
        self.edge_features = edge_features
        self.activation_function_input_mlp = ReLU()
        self.activation_function_GIN = activation_function_gin
        self.aggregation_global = aggregation_global
        self.device = device

        print(
            f"model input \n"
            f"---------- \n"
            f"input (x) channels gin block layer: {in_channels_gin_x} \n"
            f"input (edge) channels gin block layer: {in_channels_gin_edge} \n"
            f"hidden channels gin block layer: {hidden_channels_gin} \n"
            f"output channels gin block layer: {out_channels_gin} \n"
            f"---------- \n"
            f"input channels mlp output layer: {hidden_channels_global} \n"
            f"hidden channels mlp output layer: {hidden_channels_global} \n"
            f"output channels mlp output layer: {out_channels_global} \n"
            f"---------- \n"
        )

        if activation_function_mlp == "ReLU":
            self.activation_function_MLP = ReLU()
        elif activation_function_mlp == "LeakyReLU":
            self.activation_function_MLP = LeakyReLU()
        else:
            "selected activation function could not be used for the MLP, using ReLU instead"
            self.activation_function_mlp = ReLU()

        self.mlp_out = Sequential(
            Linear(hidden_channels_global, hidden_channels_global),
            BatchNorm1d(hidden_channels_global),
            self.activation_function_MLP,
            Dropout(dropout),
            Linear(hidden_channels_global, out_channels_global),
        )

        self.in_channels_gin_edge = in_channels_gin_edge
        self.in_channels_gin_x = in_channels_gin_x
        self.hidden_channels_gin = hidden_channels_gin

        # INPUT GIN BLOCK (edges)
        self.gin_MLP_layers.append(
            Sequential(
                Linear(self.in_channels_gin_edge, self.hidden_channels_gin),
                self.activation_function_input_mlp,
                Linear(self.hidden_channels_gin, self.hidden_channels_gin),
                self.activation_function_input_mlp,
                Linear(self.hidden_channels_gin, self.hidden_channels_gin),
            ).to(self.device)
        )
        # INPUT GIN BLOCK (nodes) --> see forward(..)
        self.node_embd = Sequential(
            Linear(self.in_channels_gin_x, self.hidden_channels_gin),
            self.activation_function_input_mlp,
            Linear(self.hidden_channels_gin, self.hidden_channels_gin),
            self.activation_function_input_mlp,
            Linear(self.hidden_channels_gin, self.hidden_channels_gin),
        ).to(self.device)
        

        # CORE GIN BLOCK
        for i in range(num_layers):
            self.gin_MLP_layers.append(
                GINE_layer(
                    [hidden_channels_gin, hidden_channels_gin, hidden_channels_gin],
                    train_eps=True,
                    edge_dim=edge_dim,
                    linear_learn=linear_learn,
                    activation_function=self.activation_function_GIN,
                    aggregation=aggregation_nodes_edges,
                ).to(self.device)
            )

        # OUTPUT GIN BLOCK
        self.gin_MLP_layers.append(
            Sequential(
                Linear(hidden_channels_gin, hidden_channels_gin),
                BatchNorm1d(hidden_channels_gin),
                self.activation_function_MLP,
                Dropout(dropout),
                Linear(hidden_channels_gin, hidden_channels_gin),
            ).to(self.device)
        )


    def forward(self, x, edge_index, batch, edge_attr):
        if self.aggregation_global == "add":
            aggregation = global_add_pool
        elif self.aggregation_global == "mean":
            aggregation = global_mean_pool
        elif self.aggregation_global == "max":
            aggregation = global_max_pool
        else:
            "selected aggregation function could not be used for the global aggregation, will use add aggregation instead"
            aggregation = global_add_pool

        if self.activation_function_GIN == "ReLU":
            activation_function_gin = torch.nn.functional.relu
        elif self.activation_function_GIN == "LeakyReLU":
            activation_function_gin = torch.nn.functional.leaky_relu
        else:
            "selected activation function could not be used for the forward GIN layer, using ReLU instead"
            activation_function_gin = torch.nn.functional.relu

        out_blocks = []
        start_block = 0

        # Input embeddings
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        
        edge_attr = self.gin_MLP_layers[start_block](edge_attr)
        
        MLPembd_ = MLPembd(self.in_channels_gin_x, self.hidden_channels_gin, self.activation_function_input_mlp).to(self.device)
        x = MLPembd_(x, edge_index)
        #(self.in_channels_gin_x, self.hidden_channels_gin, self.activation_function_input_mlp)(x, edge_index)

        #x = node_embd(x)
        
        # GIN layers
        for blocks_i in range(self.num_layers):
            if self.edge_features:
                x = self.gin_MLP_layers[(blocks_i)+1](x, edge_index, edge_attr)
                out_blocks.append(x)
            else:
                raise NotImplementedError

        # Global
        h = aggregation(out_blocks[0], batch)
        for i in range(self.num_layers - 1):
            h_i = aggregation(out_blocks[i + 1], batch)
            h = torch.cat((h, h_i), dim=1)
        h_pool = h

        h = self.mlp_out(h_pool)

        h_sigmoid = torch.sigmoid(h)

        return h, h_sigmoid

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        return torch.square(w).sum()
