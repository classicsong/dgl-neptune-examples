from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch as th
import os, sys
from dgl import DGLGraph
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url

sys.path.append('../../')
from utils.basic_loader import NodeClassificationDataloader

class CoraDataset(NodeClassificationDataloader):
    r"""Cora citation network dataset. Nodes mean author and edges mean citation
    relationships.
    """
    def __init__(self, device, self_loop=True, valid_ratio=0.1, test_ratio=0.2):
        super(CoraDataset, self).__init__('cora')
        self._download_data()
        self._load_raw_graph((None, "{}/cora/cora.cites".format(self._dir),'\t', [0, 1]))
        self._load_raw_label(("{}/cora/cora.content".format(self._dir),'\t', [0, -1]))
        self._load_onehot_feature(("{}/cora/cora.content".format(self._dir),'\t', [0, (1,-1)]), device)
        self._build_cora_graph(self_loop)
        self._load_node_feature(device)
        self._split_labels(device, valid_ratio, test_ratio)

        self._n_classes = len(self._labels[0].label_map)
        n_edges = self._g.number_of_edges()
        print("""----Data statistics------'
        #Edges %d
        #Classes %d
        #Train samples %d
        #Val samples %d
        #Test samples %d""" %
            (n_edges, self._n_classes,
                self._train_set[0].shape[0],
                self._valid_set[0].shape[0],
                self._test_set[0].shape[0]))

    def _download_data(self):
        self._dir = get_download_dir()
        zip_file_path='{}/{}.zip'.format(self._dir, self._name)
        download(_get_dgl_url("dataset/cora_raw.zip"), path=zip_file_path)
        extract_archive(zip_file_path,
                        '{}/{}'.format(self._dir, self._name))

    def _build_cora_graph(self, self_loop=True):
        raw_graph = self._triplets[0]
        edges = raw_graph.edges
        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                            (edges[:, 0], edges[:, 1])),
                            shape=(raw_graph.src_range, raw_graph.dst_range),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        g = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        print(g.number_of_edges())
        if self_loop:
            g.remove_edges_from(nx.selfloop_edges(g))
            g.add_edges_from(zip(g.nodes(), g.nodes()))
        g = DGLGraph(g)
        self._g = g

    def _split_labels(self, device, valid_ratio=0.1, test_ratio=0.2):
        ids, labels = self._labels[0].id_labels
        ids = th.LongTensor(ids).to(device)
        labels = th.LongTensor(labels).to(device)
        train_idx = range(140)
        valid_idx = range(200, 500)
        test_idx = range(500, 1500)

        self._test_set = (ids[test_idx], labels[test_idx])
        self._valid_set = (ids[valid_idx], labels[valid_idx])
        self._train_set = (ids[train_idx], labels[train_idx])

    @property
    def n_class(self):
        return self._n_classes