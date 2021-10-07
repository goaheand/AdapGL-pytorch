import torch
import numpy as np
from .adj_mx import get_adj


class GCNLayer(torch.nn.Module):
    """
    One Graph Convolutinal Network layer.

    Args:
        adj_num: The number of adjacent matrix.
        with_res: Whether to use residual module.

    Shape:
        - Input:
            x: :math:`(batch\_size, num\_nodes, f_{in})`
            adj: :math:`(graph\_num, num\_nodes, num\_nodes)` or `(num\_nodes, num\_nodes)`
        - Output:
            :math:`(batch\_size, num\_nodes, f_{out})`
    """
    def __init__(self, input_size, out_feature, adj_num, with_res=True):
        super(GCNLayer, self).__init__()
        self.with_res = with_res
        self.adj_num = adj_num

        self.w = torch.nn.Linear(adj_num * input_size, out_feature, bias=False)

        if with_res:
            self.w_res = torch.nn.Linear(out_feature, input_size)

    def forward(self, x, adj):
        b, node_num, _ = x.size()
        if self.adj_num == 1:
            y = torch.matmul(adj, x)
        else:
            x1 = x.unsqueeze(dim=-2)
            y = torch.matmul(adj, x1).transpose(-2, -3).reshape(b, node_num, -1)
            y = torch.relu(self.w(y))
        if self.with_res:
            y = self.w_res(y) + x
        return y


class GCN(torch.nn.Module):
    """
    Graph Convolutional Neural Network.

    Args:
        input_size: Dimension of input.
        hidden_sizes: Dimension of hidden layers (Iterable).
        step_num_out: Dimension of output.
        adj_path: The path of adjacent matrix (amx).
        device: Device to run.
        adj_type: The type of adjacent matrix, which is 'gcn' or 'cheb'. if set to 'gcn',
            :math:`A = \widetilde{D}^{-1/2} \widetilde{A} \widetilde{D}^{-1/2}`, else
            A will be computed by Chebyshev Polynomials.
        with_res: Whether to use residual module for each GCNLayer.
        **kwargs: Other keyword arguements.

    Note:
        - If adj_type is set to 'gcn', 'K=?' should be offered in '**kwargs', else it will be
        set as default value 3.
    """
    def __init__(self, input_size, hidden_sizes, step_num_out, adj_path, device,
                 adj_type='gcn', with_res=True, **kwargs):
        super(GCN, self).__init__()

        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if adj_type == 'cheb':
            self.K = kwargs.get('K', 3)

        self.adj_mx = self.build_adj_matrix(np.load(adj_path), device, adj_type=adj_type, **kwargs)
        self.gcn_list = self._build_gcn_network(input_size, hidden_sizes, step_num_out, with_res)

    def _build_gcn_network(self, input_size, hidden_sizes, step_num_out, with_res=True):
        gcn_list = torch.nn.ModuleList()

        f_in = input_size
        for hidden_size in hidden_sizes:
            gcn_list.append(GCNLayer(f_in, hidden_size, self.K, with_res))
            if not with_res:
                f_in = hidden_size
        gcn_list.append(GCNLayer(f_in, step_num_out, self.K, with_res=False))

        return gcn_list

    @staticmethod
    def build_adj_matrix(adj, device, adj_type=None, **kwargs):
        K = kwargs.get('K', 3)
        adj_mx = get_adj(adj, adj_type=adj_type)

        if adj_type != 'cheb':
            return torch.tensor(adj_mx, dtype=torch.float32, device=device)

        cheb_list = [np.eye(adj_mx.shape[0]), adj_mx]
        for _ in range(2, K):
            cheb_list.append(2 * np.dot(adj_mx, cheb_list[-1]) - cheb_list[-2])

        return torch.tensor(cheb_list, dtype=torch.float32, device=device)

    def forward(self, x):
        for gcn_layer in self.gcn_list:
            x = gcn_layer(x, self.adj_mx)
        return x.squeeze()
