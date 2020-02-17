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

class NeptuneCoraDataset(NodeClassificationDataloader):
    r"""Cora citation network dataset. Nodes mean author and edges mean citation
    relationships.

    The data is from Neptune database
    """
    def __init__(self, device, self_loop=True, valid_ratio=0.1, test_ratio=0.2):
        super(NeptuneCoraDataset, self).__init__('cora')

        # Step 1: load feature for the graph and build id mapping
        # we ignore the label row, data is stored as '~id ~label feats ...'
        self._load_onehot_feature([("~/data/cora/1581903495972/nodes/Case_Based-1.csv",',', [0, (2,0)], (1,0)),
                                   ("~/data/cora/1581903495972/nodes/Genetic_Algorithms-1.csv",',', [0, (2,0)], (1,0)),
                                   ("~/data/cora/1581903495972/nodes/Neural_Networks-1.csv",',', [0, (2,0)], (1,0)),
                                   ("~/data/cora/1581903495972/nodes/Probabilistic_Methods-1.csv",',', [0, (2,0)], (1,0)),
                                   ("~/data/cora/1581903495972/nodes/Reinforcement_Learning-1.csv",',', [0, (2,0)], (1,0)),
                                   ("~/data/cora/1581903495972/nodes/Rule_Learning-1.csv",',', [0, (2,0)], (1,0)),
                                   ("~/data/cora/1581903495972/nodes/Theory-1.csv",',', [0, (2,0)], (1,0))], device)
        # Step 2: load labels
        # we ignore the label row, data is stored as '~id ~label feats ...'
        self._load_raw_label([("~/data/cora/1581903495972/nodes/Case_Based-1.csv", ',', [0, 1], (1,0)),
                              ("~/data/cora/1581903495972/nodes/Genetic_Algorithms-1.csv",',', [0, 1], (1,0)),
                              ("~/data/cora/1581903495972/nodes/Neural_Networks-1.csv",',', [0, 1], (1,0)),
                              ("~/data/cora/1581903495972/nodes/Probabilistic_Methods-1.csv",',', [0, 1], (1,0)),
                              ("~/data/cora/1581903495972/nodes/Reinforcement_Learning-1.csv",',', [0, 1], (1,0)),
                              ("~/data/cora/1581903495972/nodes/Rule_Learning-1.csv",',', [0, 1], (1,0)),
                              ("~/data/cora/1581903495972/nodes/Theory-1.csv",',', [0, 1], (1,0))])
        # Step 3: load graph
        # we ignore the label row, data is streod as '~edge_id ~edge_label ~from ~to', we use from and to here
        self._load_raw_graph([(None, "~/data/cora/1581903495972/edges/edge-1.csv",',', [2, 3], (1,0))])
        # Step 4: build graph
        self._build_graph(self_loop, symmetric=True)
        # Step 5: load node feature
        self._load_node_feature(device)
        # Step 6: Split labels
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

    @property
    def n_class(self):
        return self._n_classes

class CoraDataset(NodeClassificationDataloader):
    r"""Cora citation network dataset. Nodes mean author and edges mean citation
    relationships.
    """
    def __init__(self, device, self_loop=True, valid_ratio=0.1, test_ratio=0.2):
        super(CoraDataset, self).__init__('cora')
        self._download_data()
        # Step 1: load feature for the graph and build id mapping
        self._load_onehot_feature([("{}/cora/cora.content".format(self._dir),'\t', [0, (1,-1)], (0,0))], device)
        # Step 2: load labels
        self._load_raw_label([("{}/cora/cora.content".format(self._dir),'\t', [0, -1], (0,0))])
        # Step 3: load graph
        self._load_raw_graph([(None, "{}/cora/cora.cites".format(self._dir),'\t', [0, 1], (0,0))])
        # Step 4: build graph
        self._build_graph(self_loop, symmetric=True)
        # Step 5: load node feature
        self._load_node_feature(device)
        # Step 6: Split labels
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