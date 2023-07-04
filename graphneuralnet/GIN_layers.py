# Building GIN layers for the graph neural network

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, LeakyReLU
from torch_geometric.nn.conv import (
    MessagePassing,
)
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.typing import OptPairTensor, Size
from torch_sparse import SparseTensor, matmul


class GIN_Conv(MessagePassing):
    def __init__(
        self,
        input_size: List[int] = [4, 2, 1],
        eps: float = 0.0,
        train_eps: bool = False,
        activation_function="ReLU",
        aggregation="add",
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggregation)
        super().__init__(**kwargs)

        self.activation_function = activation_function
        if self.activation_function == "ReLU":
            self.activation_function_nn = ReLU()
        elif self.activation_function == "LeakyReLU":
            self.activation_function_nn = LeakyReLU()
        else:
            "selected activation function could not be used for the GIN layer, using ReLU instead"
            self.activation_function_nn = ReLU()

        nn = Sequential(
            Linear(input_size[0], input_size[1]),
            BatchNorm1d(input_size[1]),
            self.activation_function_nn,
            Linear(input_size[1], input_size[2]),
            self.activation_function_nn,
        )
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


class GINE_Conv(MessagePassing):
    def __init__(
        self,
        input_size: List[int] = [4, 2, 1],
        eps: float = 0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        linear_learn=True,
        activation_function="ReLU",
        aggregation="add",
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggregation)
        super().__init__(**kwargs)

        self.activation_function = activation_function
        if self.activation_function == "ReLU":
            self.activation_function_nn = ReLU()
            self.activation_function_message = torch.nn.functional.relu
        elif self.activation_function == "LeakyReLU":
            self.activation_function_nn = LeakyReLU()
            self.activation_function_message = torch.nn.functional.leaky_relu
        else:
            "selected activation function could not be used for the GIN layer, using ReLU instead"
            self.activation_function_nn = ReLU()
            self.activation_function_message = torch.nn.functional.relu

        nn = Sequential(
            Linear(input_size[0], input_size[1]),
            BatchNorm1d(input_size[1]),
            self.activation_function_nn,
            Linear(input_size[1], input_size[2]),
            self.activation_function_nn,
        )
        self.nn = nn
        self.initial_eps = eps
        self.linear_learning = linear_learn
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, "in_features"):
                in_channels = nn.in_features
            elif hasattr(nn, "in_channels"):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")

            if self.linear_learning == True:
                self.lin = Linear(edge_dim, in_channels)
            else:
                # print("non linear layer is used for the edge embeddings (*ReLU)")
                self.lin = LeakyReLU()  # ReLU, PReLU

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None and self.linear_learning == True:
            self.lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError(
                "Node and edge feature dimensionalities do not "
                "match. Consider setting the 'edge_dim' "
                "attribute of 'GINEConv'"
            )

        # if self.lin is not None:
        # edge_attr = self.lin(edge_attr)
        # edge_attr = Linear(edge_attr.size()[1], x_j.size()[1])(
        #    edge_attr
        # )  # TODO: notimplemented error

        return self.activation_function_message(x_j + edge_attr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"
