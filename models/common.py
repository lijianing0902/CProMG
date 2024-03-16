
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor
import numpy as np

def split_tensor_by_batch(x, batch, num_graphs=None):
    if num_graphs is None:
        num_graphs = mnp.unique(batch).size
    x_split = []
    for i in range(num_graphs):
        mask = (batch == i).astype(ms.bool_)
        x_split.append(ops.boolean_mask(x, mask))
    return x_split

def concat_tensors_to_batch(x_split):
    x = ops.concatenate(x_split, 0)
    batch = ops.repeat_interleave(
        mnp.arange(len(x_split)),
        repeats=mnp.array([s.shape[0] for s in x_split]).astype(ms.int32)
    )
    return x, batch

def split_tensor_to_segments(x, segsize):
    num_segs = mnp.ceil(x.shape[0] / segsize).astype(ms.int32)
    segs = []
    for i in range(num_segs):
        segs.append(x[i * segsize: (i + 1) * segsize])
    return segs

def split_tensor_by_lengths(x, lengths):
    segs = []
    for l in lengths:
        segs.append(x[:l])
        x = x[l:]
    return segs

def batch_intersection_mask(batch, batch_filter):
    batch_filter = mnp.unique(batch_filter)
    mask = (batch.view(1, -1) == batch_filter.view(-1, 1)).any(axis=0)
    return mask




def split_tensor_by_lengths(x, lengths):
    segs = []
    for l in lengths:
        segs.append(x[:l])
        x = x[l:]
    return segs


class MultiLayerPerceptron(nn.Cell):
    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if activation == "relu":
            self.activation = ops.ReLU()
        elif activation == "sigmoid":
            self.activation = ops.Sigmoid()
        else:
            self.activation = None

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.layers = nn.CellList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Dense(self.dims[i], self.dims[i + 1]))

    def construct(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class SmoothCrossEntropyLoss(nn.Cell):
    def __init__(self, smoothing=0.0, reduction='mean'):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.log_softmax = ops.LogSoftmax(axis=-1)
        self.reduce_sum = ops.ReduceSum()
        self.reduction = reduction.lower()

        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction mode '{self.reduction}'. Use 'none', 'mean', or 'sum'.")

    def _smooth_one_hot(self, targets, n_classes):
        targets = (1.0 - self.smoothing) * targets + self.smoothing / n_classes
        return targets

    def construct(self, logits, targets):
        targets_smoothed = self._smooth_one_hot(targets, logits.shape[-1])
        log_probs = self.log_softmax(logits)
        loss = -self.reduce_sum(targets_smoothed * log_probs, axis=-1)

        if self.reduction == 'mean':
            loss = ops.reduce_mean(loss)
        elif self.reduction == 'sum':
            loss = ops.reduce_sum(loss)

        return loss


class GaussianSmearing(nn.Cell):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()

        offset = ops.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ops.square(offset[1] - offset[0]).item()
        self.offset = ops.depend(offset, ops.zeros_like(offset))

    def construct(self, dist):
        dist = dist.view(-1, 1) - ops.reshape(self.offset, (1, -1))
        return ops.exp(self.coeff * ops.square(dist))


class Gaussian(nn.Cell):
    def __init__(self, sigma):
        super(Gaussian, self).__init__()
        self.sigma = sigma
        self.exp = ops.Exp()
        self.square = ops.Square()
        self.mul = ops.Mul()
        self.div = ops.RealDiv()

    def construct(self, dist):
        return self.exp(self.div(-self.square(dist), 2 * self.sigma**2))


class ShiftedSoftplus(nn.Cell):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = ops.log(ops.scalar_to_tensor(2.0))
        self.softplus = ops.Softplus()

    def construct(self, x):
        return self.softplus(x) - self.shift



def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    batch_ctx = ops.Concat(0)((batch_protein, batch_ligand))
    sort_idx = ops.argsort()(batch_ctx)

    mask_protein = ops.Concat(0)((ops.OnesLike()(batch_protein, dtype=ms.bool_),
                                  ops.ZerosLike()(batch_ligand, dtype=ms.bool_)))[sort_idx]

    batch_ctx = ops.Gather()(batch_ctx, sort_idx)
    h_ctx = ops.Gather()(ops.Concat(0)((h_protein, h_ligand)), sort_idx)       # (N_protein+N_ligand, H)
    pos_ctx = ops.Gather()(ops.Concat(0)((pos_protein, pos_ligand)), sort_idx) # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx


def get_complete_graph(batch):
    """
    Args:
        batch:  Batch index.
    Returns:
        edge_index: (2, N_1 + N_2 + ... + N_{B-1}), where N_i is the number of nodes of the i-th graph.
        neighbors:  (B, ), number of edges per graph.
    """
    natoms = ops.ScatterAdd()(ms.ones_like(batch), batch)

    natoms_sqr = (natoms ** 2).astype(np.int32)
    num_atom_pairs = natoms_sqr.sum()
    natoms_expand = ops.Tile()(natoms, natoms_sqr)

    index_offset = ops.CumSum()(natoms) - natoms
    index_offset_expand = ops.Tile()(index_offset, natoms_sqr)

    index_sqr_offset = ops.CumSum()(natoms_sqr) - natoms_sqr
    index_sqr_offset = ops.Tile()(index_sqr_offset, natoms_sqr)

    atom_count_sqr = ops.Range()(0, num_atom_pairs).astype(np.int32) - index_sqr_offset

    index1 = (atom_count_sqr // natoms_expand).astype(np.int32) + index_offset_expand
    index2 = (atom_count_sqr % natoms_expand).astype(np.int32) + index_offset_expand
    edge_index = ops.Concat(0)((index1.reshape(1, -1), index2.reshape(1, -1)))
    mask = ops.logical_not(index1 == index2)
    edge_index = edge_index[:, mask]

    num_edges = natoms_sqr - natoms  # Number of edges per graph

    return edge_index, num_edges


def compose_context_stable(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    num_graphs = int(batch_ligand.max().asnumpy()) + 1

    batch_ctx = []
    h_ctx = []
    pos_ctx = []
    mask_protein = []

    for i in range(num_graphs):
        mask_p, mask_l = ops.Equal()(batch_protein, i), ops.Equal()(batch_ligand, i)
        batch_p, batch_l = ops.IndexSelect()(batch_protein, mask_p), ops.IndexSelect()(batch_ligand, mask_l)

        batch_ctx += [batch_p, batch_l]
        h_ctx += [ops.IndexSelect()(h_protein, mask_p), ops.IndexSelect()(h_ligand, mask_l)]
        pos_ctx += [ops.IndexSelect()(pos_protein, mask_p), ops.IndexSelect()(pos_ligand, mask_l)]
        mask_protein += [
            ops.OnesLike()(batch_p, dtype=ms.bool_),
            ops.ZerosLike()(batch_l, dtype=ms.bool_),
        ]

    batch_ctx = ops.Concat(0)(batch_ctx)
    h_ctx = ops.Concat(0)(h_ctx)
    pos_ctx = ops.Concat(0)(pos_ctx)
    mask_protein = ops.Concat(0)(mask_protein)

    return h_ctx, pos_ctx, batch_ctx, mask_protein