from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import os, sys
from dgl import DGLGraph
import dgl.backend as F

def get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class BasicGraph(object):
    r"""Basic object storing parsed graph info

        Parameters
        ----------
        edges : np.array
            Numpy array in shape of (N, 2), which means (src, dst) pairs
        is_homo : bool
            If True, graph is homo
            if False, graph is hetero
        id_mapping : dict
            Id mapping of src nodes and dst nodes
        src_name : str
            Src node type name, only for heterograph
        rel_name : str
            Relation type name, only for heterograph
        dst_name : str
            Dest node type, only for heterograph
    """
    def __init__(self, edges, id_mapping, is_homo=True, src_name=None, rel_name=None, dst_name=None):
        self._edges = edges
        self._is_homo = is_homo
        sid_map, did_map = id_mapping
        self._sid_map = sid_map
        self._did_map = did_map
        self._src_name = src_name
        self._rel_name = rel_name
        self._dst_name = dst_name

    @property
    def edges(self):
        return self._edges

    @property
    def edge_type(self):
        return (self._src_name, self._rel_name, self._dst_name)

    @property
    def is_homo(self):
        return self._is_homo

    @property
    def src_id_map(self):
        return self._sid_map

    @property
    def src_range(self):
        return len(self._sid_map)

    @property
    def dst_id_map(self):
        return self._did_map

    @property
    def dst_range(self):
        return len(self._did_map)

class BasicFeature(object):
    r"""Basic object storing parsed graph info

        Parameters
        ----------
        node_ids : np.array
            Numpy array in shape of (N, ), which means node_ids
        features : np.array
            Numpy array in shape of (N, x), which means features
        is_homo : bool
            If True, graph is homo
            if False, graph is hetero
        node_type : str
            Type name of nodes
    """
    def __init__(self, node_ids, features, is_homo=True,
                 node_type=None):
        self._node_ids = node_ids
        self._features = features
        self._is_homo = is_homo
        self._node_type = node_type

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def features(self):
        return self._features

    @property
    def is_homo(self):
        return self._is_homo

    @property
    def node_type(self):
        return self._node_type

class BasicLabel(object):
    r"""Basic object storing parsed graph info

        Parameters
        ----------
        id_labels : np.array
            Numpy array in shape of (N, 2), which means (node, label) pairs
        is_homo : bool
            If True, graph is homo
            if False, graph is hetero
        id_map : Dict
            Id mapping for nodes
        label_map : Dict
            Id mapping for labels
        node_name : str
            Type name of nodes
        label_name : str
            Type name of labels
    """
    def __init__(self, id_labels, id_map, label_map, is_homo=True,
                 node_name=None, label_name=None):

        self._id_labels = id_labels
        self._id_map = id_map
        self._label_map = label_map
        self._is_homo = is_homo
        self._node_name = node_name
        self._label_name = label_name

    @property
    def id_labels(self):
        return self._id_labels

    @property
    def label_name(self):
        return self._label_name

    @property
    def node_name(self):
        return self._node_name

    @property
    def is_homo(self):
        return self._is_homo

    @property
    def node_id_map(self):
        return self._id_map

    @property
    def label_map(self):
        return self._label_map

class NodeClassificationDataloader(object):
    r"""Basic dataset class for node classification task
    """
    def __init__(self, name):
        self._name = name
        self._id_maps = {}
        self._id_inv_maps = {}
        self._rel_maps = {}
        self._triplets = []
        self._labels = []
        self._label_map = None
        self._inv_label_map = None
        self._features = []

    def _load_raw_graph(self, graph_datas, reverse=True):
        r"""parse graph data

        Parameters
        ----------
        graph_datas : (name, file_path, separator, columns, rows)
            or List((name, file_path, separator, columns, rows)) if there are multiple files
            name :       Name of this data, can be None
            file_path :  Which file to parse
            separator :  Separator in csv
            columns: How to parse each column in csv
                column_keys is a List, with following format [(key,type),(key,type),(key,type)...] 
                or [idx, idx]
                if column_keys in format [idx, idx]
                    We donot parse csv according to column name but through column idx. The should exist 
                    only two idxes, first for src node and the second for dst node.
                    The corresponding graph is treated as homograph.
                else:
                    We will treat the graph as hetero. 
                    if only two (key, type) is provided:
                        the first is treated as src and second is treated as dst
                    else three (key, type) is provided:
                         the first is treated as src, the second is treated as relation and
                        the third is treated as dst
            rows: range of rows as features: (start:end) if end == 0, means end of row

        Return
        ------
        triplets : List
            List of BasicGraph
        id_maps : Dict
            A dictionary: type_name : id_map
        rel_maps : Dict (Optional, only in heterograph)
            A dictonary : relation type : rel_id
        """
        all_edges = []
        id_map = {} if self._id_maps.get('homo', None) is None else \
                       self._id_maps['homo']
        for graph_data in graph_datas:
            name, file_path, separator, columns, rows = graph_data
            assert isinstance(columns, list), "each edge should in order of src, relation, dst"
            if isinstance(columns[0], int):
                assert len(columns) == 2
                # homo graph
                info = pd.read_csv(file_path, sep=separator, header=None, low_memory=False)
                n_rows = info.shape[0]
                row_start, row_end = rows
                info = info.iloc[row_start:n_rows if row_end==0 else row_end, columns]
                # now parse edges, both src and dst are int64
                edges = []
                for row_val in info.to_numpy(dtype=np.int64):
                    src = row_val[0]
                    dst = row_val[1]
                    src_id = get_id(id_map, src)
                    dst_id = get_id(id_map, dst)
                    edges.append((src_id, dst_id))
                edges = np.asarray(edges, dtype=np.int64)
                all_edges.append(edges)
            else:
                pass # not impl yet

        edges = np.concatenate(all_edges)
        self._triplets.append(BasicGraph(edges, (id_map, id_map)))
        if self._id_maps.get('homo', None) is None:
            self._id_maps['homo'] = id_map

    def _load_onehot_feature(self, feature_datas, row_norm=True):
        r"""parse node feature data
        feature should be in following format
        node f1 f2 f3 f4 ... fn
        1     0  1  1  0 ...  0
        2     1  0  0  0 ...  0
        3     0  1  1  1 ...  1
        ...
        N     1  0  0  0 ...  1

        Parameters
        ----------
        feature_datas : (file_name, separator, columns, rows)
            or List((file_name, separator, columns, rows)) if there are multiple files
            file_name :  Which file to parse
            separator :  Separator in csv
            columns: column_keys is a List, with following format [(key,type),(key,type),(key,type)...]
                or [node_id, (start:end)]
                if column_keys in format [node_id, (start:end)]
                    We donot parse csv according to column name but through column idx.
                    node_id means the column idx of node_id, (start:end) means the colmun range for features,
                    if end == 0, means end of column.
                else:
                    Two or more (key, type) pairs should be provided here. Fist is treated as node_id, 
                    The others are treated as features
            rows: range of rows as features: (start:end) if end == 0, means end of row

        Return
        ------
        
        """
        feats = []
        nids = []
        id_map = {} if self._id_maps.get('homo', None) is None else \
                       self._id_maps['homo']
        # only support homo graph now
        for feature_data in feature_datas:
            file_path, separator, columns, rows = feature_data
            assert isinstance(columns, list)
            if isinstance(columns[0], int):
                assert len(columns) == 2
                assert len(rows) == 2
                # homo graph
                info = pd.read_csv(file_path, sep=separator, header=None, low_memory=False)
                n_cols = info.shape[1]
                n_rows = info.shape[0]
                row_start, row_end = rows
                col_start, col_end = columns[1]
                node_info = info.iloc[row_start:n_rows if row_end==0 else row_end,
                                      columns[0]]
                feature_info = info.iloc[row_start:n_rows if row_end==0 else row_end,
                                         col_start:n_cols if col_end==0 else col_end]
                node_ids = []
                # node id should be int64
                for nid in node_info.to_numpy(dtype=np.int64):
                    id = get_id(id_map, nid)
                    node_ids.append(id)

                node_ids = np.asarray(node_ids)
                features = feature_info.to_numpy(dtype=np.float32)
                features = sp.csr_matrix(features, dtype=np.float32)
                if row_norm:
                    features = row_normalize(features)
                features = np.array(features.todense())
                feats.append(features)
                nids.append(node_ids)
            else:
                pass #not impl yet

        features = np.concatenate(feats)
        node_ids = np.concatenate(nids)

        # sort features and node_ids
        features = features[node_ids]
        node_ids = np.arange(node_ids.shape[0])
        self._features.append(BasicFeature(node_ids, features))
        if self._id_maps.get('homo', None) is None:
            self._id_maps['homo'] = id_map

    def _load_raw_label(self, label_datas):
        r"""parse label data

        Parameters
        ----------
        label_datas : (file_name, separator, columns, rows)
            or List((file_name, separator, columns, rows)) if there are multiple files
            file_name :  Which file to parse
            separator :  Separator in csv
            columns: How to parse each column in csv
                columns is a List, with following format [(key,type),(key,type),(key,type)...]
                or [idx, idx]
                if columns in format [idx, idx]
                    We donot parse csv according to column name but through column idx.
                    There should exist only two idxes, first is treated as src, second is treated as label.
                else:
                    Two (key, type) pairs should be provided here. Fist is treated as src, 
                    Second is treated as Label.
            rows: range of rows as features: (start:end) if end == 0, means end of row

        Return
        ------
        labels : List
            A List of BasicLabel
        """
        nids = []
        nlabels = []
        id_map = {} if self._id_maps.get('homo', None) is None else \
                         self._id_maps['homo']
        label_map = {}
        for label_data in label_datas:
            file_path, separator, columns, rows = label_data
            assert isinstance(columns, list)
            if isinstance(columns[0], int):
                assert len(columns) == 2
                # homo graph
                info = pd.read_csv(file_path, sep=separator, header=None, low_memory=False)
                n_rows = info.shape[0]
                row_start, row_end = rows
                node_info = info.iloc[row_start:n_rows if row_end==0 else row_end, columns[0]]
                label_info = info.iloc[row_start:n_rows if row_end==0 else row_end, columns[1]]

                # now parse label in (id, value) pairs, id will be int64
                node_info = node_info.to_numpy(dtype=np.int64)
                label_info = label_info.to_numpy()
                pairs = []
                for idx, src in enumerate(node_info):
                    label = label_info[idx]
                    src_id = get_id(id_map, src)
                    label_id = get_id(label_map, label)
                    pairs.append((src_id, label_id))
                
                pairs = np.asarray(pairs, dtype=np.int64)
                ids = pairs[:,0]
                labels = pairs[:, 1]

                nids.append(ids)
                nlabels.append(labels)
            else:
                pass #not impl yet

        ids = np.concatenate(nids)
        labels = np.concatenate(nlabels)
        self._labels.append((BasicLabel((ids, labels), id_map, label_map)))
        self._label_map = label_map
        if self._id_maps.get('homo', None) is None:
            self._id_maps['homo'] = id_map

    def _build_graph(self, self_loop=True, symmetric=False):
        if len(self._triplets) == 1:
            raw_graph = self._triplets[0]
            edges = raw_graph.edges
            adj = sp.coo_matrix((np.ones(edges.shape[0]),
                                (edges[:, 0], edges[:, 1])),
                                shape=(raw_graph.src_range, raw_graph.dst_range),
                                dtype=np.float32)

            # build symmetric adjacency matrix
            if symmetric:
                adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            g = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
            if self_loop:
                g.remove_edges_from(nx.selfloop_edges(g))
                g.add_edges_from(zip(g.nodes(), g.nodes()))
            g = DGLGraph(g)
            self._g = g
        else:
            # (TODO xiangsx) heto graph
            assert False

    def _load_node_feature(self, device):
        if len(self._features) == 1 and self._features[0].is_homo:
            features = self._features[0]
            ft = F.tensor(features.features)
            ft = F.copy_to(ft, device)
            self._g.ndata['homo_f'] = ft
        else:
            # (TODO xiangsx) heto graph
            assert False

    def _split_labels(self, device, valid_ratio=0.1, test_ratio=0.2):
        if len(self._labels) == 1 and self._labels[0].is_homo:
            ids, labels = self._labels[0].id_labels
            ids = F.tensor(ids).to(device)
            labels = F.tensor(labels).to(device)
            num_labels = ids.shape[0]
            idx = np.arange(num_labels)
            np.random.shuffle(idx)
            train_cnt = int((1 - test_ratio) * num_labels)
            train_idx = idx[:train_cnt]
            test_idx = idx[train_cnt:]
            valid_cnt = int(valid_ratio * num_labels)
            valid_idx = train_idx[:valid_cnt]
            train_idx = train_idx[valid_cnt:]

            self._test_set = (ids[test_idx], labels[test_idx])
            self._valid_set = (ids[valid_idx], labels[valid_idx])
            self._train_set = (ids[train_idx], labels[train_idx])
        else:
            # (TODO xiangsx) heto graph
            assert False

    @property
    def test_set(self):
        return self._test_set

    @property
    def valid_set(self):
        return self._valid_set

    @property
    def train_set(self):
        return self._train_set

    @property
    def features(self):
        if len(self._features) == 1:
            return {"homo":self._g.ndata['homo_f']}
        else:
            fs = {}
            for f in self._features:
                fs[f.node_type] = f.features
            return fs
    
    @property
    def g(self):
        r"""Return DGLGraph or DGLHeteroGraph
        """
        return self._g

    def translate_node(self, node_id, ntype=None):
        if ntype is None: # homo here
            ntype = 'homo'

        if self._id_inv_maps.get('homo', None) is None:
            inv_map = {v: k for k, v in self._id_maps['homo'].items()}
            self._id_inv_maps = inv_map
        return self._id_inv_maps[node_id]

    def translate_label(self, label_id):
        if self._inv_label_map is None:
            inv_map = {v: k for k, v in self._label_map.items()}
            self._inv_label_map = inv_map
        return self._inv_label_map[label_id]