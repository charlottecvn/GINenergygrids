# Building GAT

import torch
from torch.nn import (
    Linear,
    Sequential,
    BatchNorm1d,
    ReLU,
    Dropout,
    LeakyReLU,
    Tanh,
)
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from graphnetwork.GAT_layers import GATv2Conv, MLPembd  # GATConv


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels_gat_x=4,
        in_channels_gat_edge=8,
        hidden_channels_gat=16,
        hidden_channels_global=2,
        out_channels_global=1,
        num_layers=1,
        edge_dim=16,
        dropout=0.15,
        activation_function_gat="LeakyReLU",
        activation_function_mlp="LeakyReLU",
        aggregation_global="max",
        device="cuda:o",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.gat_MLP_layers = torch.nn.ModuleList()
        self.activation_function_input_mlp = ReLU()
        self.aggregation_global = aggregation_global
        self.device = device
        self.model_name = "GAT"

        if activation_function_mlp == "ReLU":
            self.activation_function_MLP = ReLU()
        elif activation_function_mlp == "LeakyReLU":
            self.activation_function_MLP = LeakyReLU()
        elif activation_function_mlp == "tanh":
            self.activation_function_MLP = Tanh()
        else:
            "selected activation function could not be used for the MLP, using ReLU instead"
            self.activation_function_MLP = ReLU()

        if activation_function_gat == "ReLU":
            self.activation_function_GAT = torch.nn.functional.relu
        elif activation_function_gat == "LeakyReLU":
            self.activation_function_GAT = torch.nn.functional.leaky_relu
        elif activation_function_gat == "tanh":
            self.activation_function_GAT = torch.nn.functional.tanh
        else:
            "selected activation function could not be used for the forward GAT layer, using ReLU instead"
            self.activation_function_GAT = torch.nn.functional.relu

        self.mlp_out = Sequential(
            Linear(hidden_channels_global, hidden_channels_global),
            BatchNorm1d(hidden_channels_global),
            self.activation_function_MLP,
            Dropout(dropout),
            Linear(hidden_channels_global, out_channels_global),
        )

        self.in_channels_gat_x = in_channels_gat_x
        self.in_channels_gat_edge = in_channels_gat_edge
        self.hidden_channels_gat = hidden_channels_gat

        # INPUT GAT BLOCK (edges)
        self.gat_MLP_layers.append(
            Sequential(
                Linear(self.in_channels_gat_edge, self.hidden_channels_gat),
                self.activation_function_input_mlp,
                Linear(self.hidden_channels_gat, self.hidden_channels_gat),
                self.activation_function_input_mlp,
                Linear(self.hidden_channels_gat, self.hidden_channels_gat),
            ).to(self.device)
        )

        # INPUT GAT BLOCK (nodes)
        self.node_embd = Sequential(
            Linear(self.in_channels_gat_x, self.hidden_channels_gat),
            self.activation_function_input_mlp,
            Linear(self.hidden_channels_gat, self.hidden_channels_gat),
            self.activation_function_input_mlp,
            Linear(self.hidden_channels_gat, self.hidden_channels_gat),
        ).to(self.device)

        # CORE GAT BLOCK
        for i in range(num_layers):
            self.gat_MLP_layers.append(
                GATv2Conv(
                    in_channels=hidden_channels_gat,
                    out_channels=hidden_channels_gat,
                    edge_dim=edge_dim,
                ).to(self.device)
            )

        # OUTPUT GAT BLOCK
        self.gat_MLP_layers.append(
            Sequential(
                Linear(hidden_channels_gat, hidden_channels_gat),
                BatchNorm1d(hidden_channels_gat),
                self.activation_function_MLP,
                Dropout(dropout),
                Linear(hidden_channels_gat, hidden_channels_gat),
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

        out_blocks = []

        edge_index = edge_index.to(self.device)

        # embeddings (nodes)
        x = x.to(self.device)
        MLPembd_ = MLPembd(
            self.in_channels_gat_x,
            self.hidden_channels_gat,
            self.activation_function_input_mlp,
        ).to(self.device)
        x = MLPembd_(x, edge_index)

        # embeddings (edges)
        edge_attr = edge_attr.to(self.device)
        edge_attr = self.gat_MLP_layers[0](edge_attr)

        # GAT layers
        for blocks_i in range(self.num_layers):
            x = self.gat_MLP_layers[(blocks_i) + 1](x, edge_index, edge_attr)
            out_blocks.append(x)

        # merge layers
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
