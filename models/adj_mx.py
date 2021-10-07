import numpy as np


def normalized_laplacian(adj):
    """
    :math:`I - D^(-1/2)AD^(-1/2)`
    """
    N = adj.shape[0]
    I = np.eye(N)
    d = adj.sum(axis=-1) ** (-0.5)
    d[np.isinf(d)] = 0.
    return I - d.reshape(-1, 1) * adj * d


def laplacian(adj):
    """
    :math:`D - A`
    """
    d = np.diag(adj.sum(axis=-1))
    return d - adj


def gcn(adj):
    """
    :math:`\widetilde{D}^{-1/2} \widetilde{A} \widetilde{D}^{-1/2}`
    :math:`\widetilde{A} = I + A`
    """
    adj = random_walk(adj)
    N = adj.shape[0]
    adj += np.eye(N)
    d = adj.sum(axis=-1) ** (-0.5)
    d[np.isinf(d)] = 0.
    return d.reshape(-1, 1) * adj * d


def random_walk_laplacian(adj):
    """
    :math:`I - D^{-1}A`
    """
    d = adj.sum(axis=-1) ** (-1)
    d[np.isinf(d)] = 0.
    return np.eye(adj.shape[0]) - d.reshape(-1, 1) * adj


def random_walk(adj):
    """
    :math:`D^{-1}A`
    """
    # adj += np.eye(adj.shape[0])
    d = adj.sum(axis=-1) ** (-1)
    d[np.isinf(d)] = 0.
    return d.reshape(-1, 1) * adj


def cheb(adj):
    """
    :math:`\frac(2){\lambda_{max}}L - I`
    :math:`L = I - D^(-1/2)AD^(-1/2)`
    """
    N = adj.shape[0]
    I = np.eye(N)
    L = normalized_laplacian(adj)
    max_eig_value = abs(np.linalg.eigvals(L)).max()
    return 2 / max_eig_value * L - I


def get_adj(adj, adj_type):
    adj_types = {
        'normalized_laplacian': normalized_laplacian,
        'laplacian': laplacian,
        'gcn': gcn,
        'random_walk_laplacian': random_walk_laplacian,
        'random_walk': random_walk,
        'cheb': cheb
    }
    func = adj_types.get(adj_type, lambda: "Invalid adj_type!")
    return func(adj)
