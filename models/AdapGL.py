import math
import torch
from .ASTGCN import TemporalAttention
from .DCRNN import GeneralDCRNN, DCGRULayer


class GraphConv(torch.nn.Module):
    r"""
    Graph Convolution with self feature modeling.

    Args:
        f_in: input size.
        num_cheb_filter: output size.
        conv_type: 
            gcn: :math:`AHW`,
            cheb: :math:``T_k(A)HW`.
        activation: default relu.
    """
    def __init__(self, f_in, num_cheb_filter, conv_type=None, **kwargs):
        super(GraphConv, self).__init__()
        self.K = kwargs.get('K', 3) if conv_type == 'cheb' else 1
        self.with_self = kwargs.get('with_self', True)
        self.w_conv = torch.nn.Linear(f_in * self.K, num_cheb_filter, bias=False)
        if self.with_self:
            self.w_self = torch.nn.Linear(f_in, num_cheb_filter)
        self.conv_type = conv_type
        self.activation = kwargs.get('activation', torch.relu)

    def cheb_conv(self, x, adj_mx):
        bs, num_nodes, _ = x.size()

        if adj_mx.dim() == 3:
            h = x.unsqueeze(dim=1)
            h = torch.matmul(adj_mx, h).transpose(1, 2).reshape(bs, num_nodes, -1)
        else:
            h_list = [x, torch.matmul(adj_mx, x)]
            for _ in range(2, self.K):
                h_list.append(2 * torch.matmul(adj_mx, h_list[-1]) - h_list[-2])
            h = torch.cat(h_list, dim=-1)

        h = self.w_conv(h)
        if self.with_self:
            h += self.w_self(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def gcn_conv(self, x, adj_mx):
        h = torch.matmul(adj_mx, x)
        h = self.w_conv(h)
        if self.with_self:
            h += self.w_self(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def forward(self, x, adj_mx):
        self.conv_func = self.cheb_conv if self.conv_type == 'cheb' else self.gcn_conv
        return self.conv_func(x, adj_mx)


class GraphLearn(torch.nn.Module):
    """
    Graph Learning Modoel for AdapGL.

    Args:
        num_nodes: The number of nodes.
        init_feature_num: The initial feature number (< num_nodes).
    """
    def __init__(self, num_nodes, init_feature_num):
        super(GraphLearn, self).__init__()
        self.epsilon = 1 / num_nodes * 0.5
        self.beta = torch.nn.Parameter(
            torch.rand(num_nodes, dtype=torch.float32),
            requires_grad=True
        )

        self.w1 = torch.nn.Parameter(
            torch.zeros((num_nodes, init_feature_num), dtype=torch.float32),
            requires_grad=True
        )
        self.w2 = torch.nn.Parameter(
            torch.zeros((num_nodes, init_feature_num), dtype=torch.float32),
            requires_grad=True
        )

        self.attn = torch.nn.Conv2d(2, 1, kernel_size=1)

        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, adj_mx):
        new_adj_mx = torch.mm(self.w1, self.w2.T) - torch.mm(self.w2, self.w1.T)
        new_adj_mx = torch.relu(new_adj_mx + torch.diag(self.beta))
        attn = torch.sigmoid(self.attn(torch.stack((new_adj_mx, adj_mx), dim=0).unsqueeze(dim=0)).squeeze())
        new_adj_mx = attn * new_adj_mx + (1. - attn) * adj_mx
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = torch.relu(d.view(-1, 1) * new_adj_mx * d - self.epsilon)
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = d.view(-1, 1) * new_adj_mx * d
        return new_adj_mx


class ChannelAttention(torch.nn.Module):
    def __init__(self, c_in):
        super(ChannelAttention, self).__init__()
        self.r = 0.5
        hidden_size = int(c_in / self.r)
        self.w1 = torch.nn.Linear(c_in, hidden_size, bias=False)
        self.w2 = torch.nn.Linear(hidden_size, c_in, bias=False)

    def forward(self, x):
        y = x.mean(dim=(-1, -2))
        y = torch.sigmoid(self.w2(torch.relu(self.w1(y))))
        return y.unsqueeze(dim=-1)


class AdapGLBlockT(torch.nn.Module):
    r""" One AdapGL block of T-GCN.

    Args:
        c_in: Nunber of time_steps.
        f_in: Number of input features.
        rnn_hidden_size: Hidden size of GRU.
        rnn_layer_num: Layer number if GRU.
        num_cheb_filter: hidden size of chebyshev graph convolution.
        num_nodes: The number of nodes.
        conv_type: The type for graph convolution:
            gcn: :math:`AHW`,
            cheb: :math:``T_k(A)HW`.

    Shape:
        - Input:
            x: :math:`(batch\_size, c_in, num\_nodes, f_in)` if batch_first is True, else
                :math:`(c_in, batch\_size, num\_nodes, f_in)`
            adj_mx: :math:`(num\_graph, num\_nodes, num\_nodes)` or :math:`(num\_nodes,num\_nodes)`
                for conv_type 'cheb', :math:`(num\_nodes, num\_nodes)` for conv_type 'gcn'.
        - Output: :math:`(batch\_size, c_in, num\_nodes, num_cheb_filter)` if batch_first is True,
            else :math:`(c_in, batch\_size, num\_nodes, rnn_hidden_size)`.
    """
    def __init__(self, c_in, f_in, rnn_hidden_size, rnn_layer_num, num_cheb_filter, conv_type,
                 batch_first=True, with_res=False, K=3):
        super(AdapGLBlockT, self).__init__()
        if with_res:
            assert rnn_hidden_size == num_cheb_filter, "hidden_size of rnn and ChebConv should be the same."
        self.with_res = with_res
        self.batch_first = batch_first
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layer_num = rnn_layer_num

        self.rnn = DCGRULayer(num_cheb_filter, rnn_hidden_size, rnn_layer_num, use_gc=False)
        self.temporal_att = ChannelAttention(c_in)

        self.graph_conv_1 = GraphConv(f_in, num_cheb_filter // 2, conv_type, K=K, activation=None,
                                      with_self=False)
        self.graph_conv_2 = GraphConv(f_in, num_cheb_filter // 2, conv_type, K=K, activation=None,
                                      with_self=False)

        self.layer_norm = torch.nn.LayerNorm(num_cheb_filter)

    def graph_conv(self, x, adj_mx):
        b, c, n_d, f_in = x.size()
        h1 = self.graph_conv_1(x.reshape(-1, n_d, f_in), adj_mx).reshape(b, c, n_d, -1)
        h2 = self.graph_conv_2(x.reshape(-1, n_d, f_in), adj_mx.T).reshape(b, c, n_d, -1)
        h = torch.cat((h1, h2), dim=-1)
        return h

    def recursive_passing(self, x):
        if self.batch_first:
            x = x.transpose(0, 1)

        h_size = [self.rnn_layer_num] + list(x.size()[1: -1]) + [self.rnn_hidden_size]
        hx = torch.zeros(h_size, dtype=torch.float32, device=x.device)

        hy = []
        for t in range(x.size(0)):
            h, hx = self.rnn(x[t], hx, adj_mx_list=None)
            hy.append(h)
        hy = torch.stack(hy).transpose(0, 1)
        return hy

    def forward(self, x, adj_mx):
        h = torch.relu(self.graph_conv(x, adj_mx))
        h = self.recursive_passing(h)

        b, c, n, _ = h.size()
        h_tat = self.temporal_att(h)
        h = (h.view(b, c, -1) * h_tat).reshape(b, c, n, -1)
        h = self.layer_norm(h)
        if not self.batch_first:
            h = h.transpose(0, 1)

        return h


class AdapGLBlockA(torch.nn.Module):
    def __init__(self, c_in, f_in, num_nodes, num_cheb_filter, num_time_filter, kernel_size,
                 conv_type, K=3):
        super(AdapGLBlockA, self).__init__()

        self.padding = (kernel_size - 1) // 2
        self.graph_conv_p = GraphConv(f_in, num_cheb_filter // 2, conv_type=conv_type,
                                      K=K, activation=None, with_self=False)
        self.graph_conv_n = GraphConv(f_in, num_cheb_filter // 2, conv_type=conv_type,
                                      K=K, activation=None, with_self=False)
        self.temporal_att = TemporalAttention(num_nodes, f_in, c_in)

        self.time_conv = torch.nn.Conv2d(
            in_channels=num_cheb_filter,
            out_channels=num_time_filter,
            kernel_size=(1, kernel_size),
            padding=(0, self.padding)
        )

        self.residual_conv = torch.nn.Conv2d(
            in_channels=f_in,
            out_channels=num_time_filter,
            kernel_size=(1, 1)
        )

        self.ln = torch.nn.LayerNorm(num_time_filter)

    def forward(self, x, adj_mx):
        b, c, n_d, f = x.size()

        temporal_att = self.temporal_att(x)
        x_tat = (torch.matmul(temporal_att, x.reshape(b, c, -1)).reshape(b, c, n_d, f))
        hp = self.graph_conv_p(x_tat.reshape(-1, n_d, f), adj_mx)
        hn = self.graph_conv_n(x_tat.reshape(-1, n_d, f), adj_mx.T)
        h = torch.relu(torch.cat((hp, hn), dim=-1).reshape(b, c, n_d, -1))

        h = self.time_conv(h.transpose(1, 3)).transpose(1, 3)
        h_res = self.residual_conv(x.transpose(1, 3)).transpose(1, 3)

        h = torch.relu(h + h_res)
        return self.ln(h)


class AdapGLA(torch.nn.Module):
    """
    Attention based Graph Learning Neural Network.

    Args:
        num_block: The number of AGLNBlock.
        num_nodes: The number of nodes.
        step_num_out: Output Channels (step_num_out).
        step_num_in: Nunber of time_steps.
        input_size: Number of input features.
        num_che_filter: hidden size of chebyshev graph convolution.
        K: The order of Chebyshev polymials.
        conv_type: The type for graph convolution:
            gcn: :math:`AHW`,
            cheb: :math:``T_k(A)HW`.
    """
    def __init__(self, **kwargs):
        super(AdapGLA, self).__init__()

        num_block = kwargs.get('num_block', 2)
        num_nodes = kwargs.get('num_nodes', None)
        c_in = kwargs.get('step_num_in', 12)
        c_out = kwargs.get('step_num_out', 12)
        f_in = kwargs.get('input_size', 1)
        kernel_size = kwargs.get('kernel_size', 3)
        num_time_filter = kwargs.get('num_time_filter', 64)
        num_cheb_filter = kwargs.get('num_cheb_filter', 64)
        conv_type = kwargs.get('conv_type', 'gcn')
        K = kwargs.get('K', 1)

        activation = kwargs.get('activation', 'relu')
        activation = getattr(torch, activation)

        self.block_list = torch.nn.ModuleList()
        for i in range(num_block):
            temp_h = f_in if i == 0 else num_time_filter
            self.block_list.append(AdapGLBlockA(
                c_in, temp_h, num_nodes, num_cheb_filter,
                num_time_filter, kernel_size, conv_type, K=K
            ))

        self.final_conv = torch.nn.Conv2d(c_in, c_out, (1, num_time_filter))

    def forward(self, x, adj_mx):
        h = x
        for net_block in self.block_list:
            h = net_block(h, adj_mx)
        h = self.final_conv(h).squeeze(dim=-1)
        return h

    def __str__(self):
        return 'AdapGLA'


class AdapGLD(GeneralDCRNN):
    def __init__(self, **kwargs):
        super(AdapGLD, self).__init__(**kwargs)

    def forward(self, x, adj_mx, labels=None, batches_seen=None):
        adj_mx_list = (adj_mx, adj_mx.T)
        return super().forward(x, adj_mx_list, labels, batches_seen)

    def __str__(self):
        return 'AdapGLD'


class AdapGLT(torch.nn.Module):
    def __init__(self, **kwargs):
        super(AdapGLT, self).__init__()

        num_block = kwargs.get('num_block', 2)
        f_in = kwargs.get('input_size', 1)
        c_in = kwargs.get('step_num_in', 12)
        c_out = kwargs.get('step_num_out', 12)
        rnn_hidden_size = kwargs.get('rnn_hidden_size', 64)
        rnn_layer_num = kwargs.get('rnn_layer_num', 1)
        num_cheb_filter = kwargs.get('num_cheb_filter', 64)
        batch_first = kwargs.get('batch_first', True)
        K = kwargs.get('K', 3)
        conv_type = kwargs.get('conv_type', 'cheb')
        with_res = kwargs.get('with_res', False)

        sel_func = lambda x: f_in if x == 0 else rnn_hidden_size
        self.module_list = torch.nn.ModuleList([AdapGLBlockT(
            c_in, sel_func(i), rnn_hidden_size, rnn_layer_num, num_cheb_filter,
            conv_type, batch_first, with_res, K) for i in range(num_block)
        ])

        self.final_conv = torch.nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=(1, rnn_hidden_size)
        )

    def forward(self, x, adj_mx):
        h = x
        for block in self.module_list:
            h = block(h, adj_mx)
        h = self.final_conv(h).squeeze()
        if x.size(0) == 1:
            h = h.unsqueeze(dim=0)
        return h

    def __str__(self):
        return 'AdapGLT'
