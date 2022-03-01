from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class my_MLP_GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        att_in_channels: int,
        att_out_channels: int,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_in_channels = att_in_channels
        self.att_out_channels = att_out_channels
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights

        self.att_in = Linear(att_in_channels, att_out_channels, bias=bias,
                            weight_initializer='glorot') 

        self.att_out = Linear(att_out_channels, 1, bias=False,
                            weight_initializer='glorot') 

        self.lin = Linear(in_channels, out_channels, bias=bias,
                            weight_initializer='glorot')

        self._alpha = None
        self._pair_pred = None

        self.reset_parameters()

    def reset_parameters(self):
        self.att_in.reset_parameters()
        self.att_out.reset_parameters()
        self.lin.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_info: bool = None):
        
        C = self.out_channels

        x_: OptTensor = None
            
        assert x.dim() == 2
        x_ = self.lin(x).view(-1, C)

        assert x_ is not None

        if self.add_self_loops:
            num_nodes = x_.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=x_, size=None)

        alpha = self._alpha
        pair_pred = self._pair_pred
        self._alpha = None
        self._pair_pred = None

        out = out.mean(dim=1)

        if isinstance(return_attention_info, bool):
            assert alpha is not None
            assert pair_pred is not None
            return out, (edge_index, alpha), pair_pred
        else:
            return out


    def message(self, x_j: Tensor, x_i: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        
        tmp = torch.cat([x_i, x_j], dim=1)
        tmp = self.att_in(tmp)
        tmp = F.leaky_relu(tmp, self.negative_slope)
        tmp = self.att_out(tmp)
        self._pair_pred = tmp
        alpha = softmax(tmp, index, ptr, size_i)
        self._alpha = alpha

        return x_j * alpha

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
