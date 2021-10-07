import torch
import numpy as np
from .adj_mx import get_adj


class DCGRUCell(torch.nn.Module):
    """
    A GRU cell with graph convolution.

    Args:
        input_size: Size of input.
        rnn_hidden_size: Hidden size of GRU cell.
        use_gc: Whether to use graph convolution instead of linear layer. If set to
            True, use graph convolution, else linear layer.
        activation: The activation function to use.
        **kwargs: other keyword arguments.

    Notes:
        - If use_gc is True, 'graph_num=?' should be offered for the number of graphs.
    """
    def __init__(self, input_size, rnn_hidden_size, use_gc=True, activation='tanh', **kwargs):
        super(DCGRUCell, self).__init__()
        self.in_size = input_size
        self.h_size = rnn_hidden_size
        self.use_gc = use_gc
        self.activation = getattr(torch, activation, torch.tanh)

        if use_gc:
            assert 'graph_num' in kwargs, 'graph_num is not set !'
            self.graph_num = kwargs.get('graph_num')
            self.K = kwargs.get('K', 3)

        gw_size = self.in_size + self.h_size
        if use_gc:
            gw_size *= self.graph_num * (self.K - 1) + 1

        self.gate_weight = torch.nn.Linear(gw_size, self.h_size * 2)
        self.cell_weight = torch.nn.Linear(gw_size, self.h_size)

    def _gconv(self, adj_mx_list, x):
        """
        Achieve :math:`AH`.

        :param adj_mx_list: The list of adjacent matrix.
        :param x: Input with shape :math:`(batch_size, num_nodes, F_{in})`
        """
        h = [x]
        for adj_mx in adj_mx_list:
            tk_2, tk_1 = x, torch.matmul(adj_mx, x)
            h.append(tk_1)
            for _ in range(2, self.K):
                tk_2, tk_1 = tk_1, 2 * torch.matmul(adj_mx, tk_1) - tk_2
                h.append(tk_1)
        return torch.cat(h, dim=-1)

    def forward(self, x, hx, adj_mx_list=None):
        """
        Gated recurrent unit (GRU) with Graph Convolution.

        :param x: (B, num_nodes, input_size) if use_gc else (B, input_dim)
        :param hx: (B, num_nodes, rnn_hidden_size) if use_gc else (B, rnn_hidden_size)
        :param adj_mx_list: (graph_num, num_nodes, num_nodes)
        :return
        - Output: The same shape as hx.
        """
        h = torch.cat((x, hx), dim=-1)
        if self.use_gc:
            h = self._gconv(adj_mx_list, h)
        h = torch.sigmoid(self.gate_weight(h))
        r, z = torch.split(tensor=h, split_size_or_sections=self.h_size, dim=-1)

        c = torch.cat((x, r * hx), dim=-1)
        if self.use_gc:
            c = self._gconv(adj_mx_list, c)
        c = self.activation(self.cell_weight(c))

        hx = z * hx + (1 - z) * c
        return hx


class DCGRULayer(torch.nn.Module):
    """
    A GRU layer with one or more stacks.

    Args:
        rnn_layer_num: The stacked number of one GRU layer.
    """
    def __init__(self, input_size, rnn_hidden_size, rnn_layer_num, use_gc=True, **kwargs):
        super(DCGRULayer, self).__init__()
        self.h_size = rnn_hidden_size

        self.dcgru_layers = torch.nn.ModuleList(
            [DCGRUCell(input_size, rnn_hidden_size, use_gc, **kwargs)]
        )
        for _ in range(1, rnn_layer_num):
            self.dcgru_layers.append(
                DCGRUCell(rnn_hidden_size, rnn_hidden_size, use_gc, **kwargs)
            )

    def forward(self, x, hx, adj_mx_list):
        hidden_states = []

        h = x
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            h = dcgru_layer(h, hx[layer_num], adj_mx_list)
            hidden_states.append(h)

        return h, torch.stack(hidden_states)


class GeneralDCRNN(torch.nn.Module):
    """ DCRNN network. """
    def __init__(self, **kwargs):
        super(GeneralDCRNN, self).__init__()

        self.input_size = kwargs.get('input_size', 1)
        self.output_size = 1
        self.rnn_hidden_size = kwargs.get('rnn_hidden_size', 32)
        self.rnn_layer_num = kwargs.get('rnn_layer_num', 1)
        self.use_gc = kwargs.get('use_gc', True)
        self.step_num_out = kwargs.get('step_num_out', 12)
        self.batch_first = kwargs.get('batch_first', False)
        self.cl_decay_steps = kwargs.get('cl_decay_steps', 2000)
        self.use_curriculum_lr = kwargs.get('use_curriculum_learning', True)
        self.activation = kwargs.get('activation', 'tanh')
        device_name = kwargs.get('device', 'cuda:0')
        self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

        K = kwargs.get('K', 3)
        graph_num = kwargs.get('graph_num', 2)
        cell_kwargs = {'graph_num': graph_num, 'K': K} if self.use_gc else {}
        self.encoder_layer = DCGRULayer(
            input_size=self.input_size,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_layer_num=self.rnn_layer_num,
            use_gc=self.use_gc,
            activation=self.activation,
            **cell_kwargs
        )
        self.decoder_layer = DCGRULayer(
            input_size=self.output_size,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_layer_num=self.rnn_layer_num,
            use_gc=self.use_gc,
            activation=self.activation,
            **cell_kwargs
        )

        self.fc_layers = torch.nn.ModuleList([
            torch.nn.Linear(self.rnn_hidden_size, self.output_size) for _ in range(self.step_num_out)
        ])

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps +
                                      np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, x, adj_mx_list=None):
        """
        encoder forward pass on t time steps.

        :param x: (seq_len, batch_size, num_nodes, input_dim) if use_gc else
            (seq_len, batch_size, input_dim), if batch_first is True, the first
            dimension will be batch_size, then seq_len.
        :param adj_mx_list: List of adjacent matrix, if use_gc is False, use None instead.
        :return: hx: (rnn_layer_num, batch_size, num_nodes, rnn_hidden_size) if use_gc
            else (runn_layer_num, batch_size, rnn_hidden_size).
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        h_size = [self.rnn_layer_num] + list(x.size()[1: -1]) + [self.rnn_hidden_size]
        hx = torch.zeros(h_size, dtype=torch.float32, device=self.device)

        for t in range(x.size(0)):
            _, hx = self.encoder_layer(x[t], hx, adj_mx_list)

        return hx

    def decoder(self, hx, adj_mx_list=None, labels=None, batches_seen=None):
        """
        Decoder forward pass on future 'step_num_out' time steps.

        :param hx: Hidden state from encoder. The shape is (rnn_layer_num, batch_size,
            num_nodes, rnn_hidden_size).
        :param adj_mx_list: List of Adjacent matrix, if use_gc is False, use None instead.
        :param labels: The output of the network. (Just used for training.)
        :param batches_seen: Global step of training. (Just used for training.)

        :return output: Output of the network. The shape is (step_num_out, batch_size,
            self.num_nodes), if batch_first is True, the first dimension will be
            batch_size, then step_num_out. Furthermore, if step_num_out == 1, the output shape
            is (batch_size, self.num_nodes).

        Notes:
            - Only if 'user_curriculum_learning' is True, that params 'labels' and '
            batch_first' can be used.
        """
        x_size = list(hx.size()[1: -1]) + [self.output_size]
        x = torch.zeros(x_size, dtype=torch.float32, device=self.device)

        if labels is not None:
            labels = labels.unsqueeze(dim=-1)
            if self.step_num_out > 1 and self.batch_first:
                labels = labels.transpose(0, 1)
            elif self.step_num_out == 1:
                labels = labels.unsqueeze(dim=0)

        y = []
        for t in range(self.step_num_out):
            x, hx = self.decoder_layer(x, hx, adj_mx_list)
            x = self.fc_layers[t](x)
            y.append(x)
            if self.training and self.use_curriculum_lr:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    x = labels[t]

        y = torch.stack(y).squeeze(dim=-1)
        if y.size(0) == 1:
            y = y.squeeze(dim=0)
        if self.batch_first and self.step_num_out > 1:
            y = y.transpose(0, 1)

        return y

    def forward(self, x, adj_mx_list, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param x: (seq_len, batch_size, num_nodes, input_dim) if use_gc else
            (seq_len, batch_size, input_dim), if batch_first is True, the first
            dimension will be batch_size, then seq_len.
        :param adj_mx_list: List of adjacent matrixes.
        :param labels: The output of the network. (Just used for training.)
        :param batches_seen: Global step of training. (Just used for training.)

        :return output: Output of the network. The shape is (step_num_out, batch_size,
            self.num_nodes), if batch_first is True, the first dimension will be
            batch_size, then step_num_out. Furthermore, if step_num_out == 1, the output shape
            is (batch_size, self.num_nodes).

        Notes:
            - Only if 'use_curriculum_learning' is True, that params 'labels' and '
            batch_first' can be used.
        """
        encoder_hidden_state = self.encoder(x, adj_mx_list)
        outputs = self.decoder(encoder_hidden_state, adj_mx_list, labels, batches_seen)
        return outputs


class DCRNN(GeneralDCRNN):
    def __init__(self, **kwargs):
        super(DCRNN, self).__init__(**kwargs)

        adj_path = kwargs.get('adj_path', None)
        adj_type = kwargs.get('adj_type', None)
        if adj_path is None:
            raise ValueError('The path of adjacent matrix should be offered!')
        self.adj_mx_list = self._load_adj(adj_path, adj_type, self.device)

    @staticmethod
    def _load_adj(adj_path: str, adj_type: str, device, **kwargs):
        adj_paths = adj_path.strip().split(',')

        adj_mx_list = []
        for adj_mx_path in adj_paths:
            adj = np.load(adj_mx_path.strip())
            adj_mx = torch.tensor(get_adj(adj, adj_type), dtype=torch.float32, device=device)
            adj_mx_list.append(adj_mx)

            adj_mx = torch.tensor(get_adj(adj.T, adj_type), dtype=torch.float32, device=device)
            adj_mx_list.append(adj_mx)

        return adj_mx_list

    def forward(self, x, labels=None, batches_seen=None):
        return super().forward(x, self.adj_mx_list, labels, batches_seen)
