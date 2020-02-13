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
        self._rel_maps = {}
        self._triplets = []
        self._labels = []
        self._label_map = {}
        self._features = []

    def _load_raw_graph(self, graph_data, reverse=True):
        r"""parse graph data

        Parameters
        ----------
        graph_data : (name, file_path, separator, column_keys)
            or List((name, file_path, separator, column_keys)) if there are multiple files
            name :       Name of this data, can be None
            file_path :  Which file to parse
            separator :  Separator in csv
            column_keys: How to parse each column in csv
                column_keys is a List, with following format [(key,type),(key,type),(key,type)...] 
                or [idx, idx]
                if column_keys in format [idx, idx]
                    The csv donot have column name, it is parsed according to column idx. The should exist 
                    only two idxes, first for src node and the second for dst node.
                    The corresponding graph is treated as homograph.
                else:
                    We will treat the graph as hetero. 
                    if only two (key, type) is provided:
                        the first is treated as src and second is treated as dst
                    else three (key, type) is provided:
                         the first is treated as src, the second is treated as relation and
                        the third is treated as dst

        Return
        ------
        triplets : List
            List of BasicGraph
        id_maps : Dict
            A dictionary: type_name : id_map
        rel_maps : Dict (Optional, only in heterograph)
            A dictonary : relation type : rel_id
        """
        if isinstance(graph_data, list):
            for d in graph_data:
                self._parse_graph_data(d)
        else:
            name, file_path, separator, column_keys = graph_data
            assert isinstance(column_keys, list), "each edge should in order of src, relation, dst"
            if isinstance(column_keys[0], int):
                assert len(column_keys) == 2
                # homo graph
                info = pd.read_csv(file_path, sep=separator, header=None)
                info = info.iloc[:,column_keys]

                # now parse edges
                id_map = {}
                edges = []
                for _, row_val in info.iterrows():
                    src, dst = row_val
                    src_id = get_id(id_map, src)
                    dst_id = get_id(id_map, dst)
                    edges.append((src_id, dst_id))
                edges = np.asarray(edges, dtype=np.int32)
                self._triplets.append(BasicGraph(edges, (id_map, id_map)))
                self._id_maps['homo' if name is None else name] = id_map
            else:
                names = [key for (key, dtype) in column_keys]
                dtypes = {key : dtype for (key, dtype) in column_keys}
                # hetero graph
                assert len(column_keys) <= 3, "Each edge is in format of src, relation, dst"
                assert len(column_keys) > 1

                info = pd.read_csv(file_path, sep=separator, header=None,
                                   names=names,
                                   dtype=dtypes)

                if len(column_keys) == 2:
                    src_id_map = {} if self._id_maps.get(names[0], None) is None else \
                                    self._id_maps[names[0]]
                    dst_id_map = {} if self._id_maps.get(names[1], None) is None else \
                                    self._id_maps[names[1]]
                    edges = []
                    for _, row_val in info.iterrows():
                        src, dst = row_val
                        src_id = get_id(src_id_map, src)
                        dst_id = get_id(dst_id_map, dst)
                        edges.append((src_id, dst_id))
                    # no reverse in homograph
                    edges = np.asarray(edges, dtype=np.int32)
                    self._triplets.append(BasicGraph(edges,
                                                     (src_id_map, dst_id_map),
                                                     False,
                                                     name[0],
                                                     'none' if name is None else name,
                                                     name[1]))
                    if self._id_maps.get(names[0], None) is None:
                        self._id_maps[names[0]] = src_id_map

                    if self._id_maps.get(names[1], None) is None:
                        self._id_maps[names[1]] = dst_id_map
                else:
                    src_id_map = {} if self._id_maps.get(names[0], None) is None else \
                                    self._id_maps[names[0]]
                    dst_id_map = {} if self._id_maps.get(names[2], None) is None else \
                                    self._id_maps[names[2]]
                    rel_id_map = {} if self._rel_maps.get(names[1], None) is None else \
                                    self._rel_maps[names[1]]
                    graphs = {}
                    for _, row_val in info.iterrows():
                        src, rel, dst = row_val
                        src_id = get_id(src_id_map, src)
                        dst_id = get_id(dst_id_map, dst)
                        rel_id = get_id(rel_id_map, rel)
                        if graphs.get(rel, None) is None:
                            graphs[rel] = [(src_id, dst_id)]
                        else:
                            graphs[rel].append((src_id, dst_id))
                    for rel, edges in graphs.items:
                        edges = np.asarray(edges, dtype=np.int32)
                        self._triplets.append(BasicGraph(edges,
                                                         (src_id_map, dst_id_map),
                                                         False,
                                                         name[0],
                                                         rel,
                                                         name[2]))
                        if reverse is True:
                            rel = "rev-{}".format(rel)
                            rel_id = get_id(rel_id_map, rel)
                            edges[:,[0,1]] = edges[:[1,0]]
                            self._triplets.append(BasicGraph(edges,
                                                             (dst_id_map, src_id_map),
                                                             False,
                                                             name[2],
                                                             rel,
                                                             name[0]))
           
                    if self._id_maps.get(names[0], None) is None:
                        self._id_maps[names[0]] = src_id_map

                    if self._id_maps.get(names[2], None) is None:
                        self._id_maps[names[2]] = dst_id_map

                    if self._rel_maps.get(names[1], None) is None:
                        self._rel_maps[names[1]] = rel_id_map

    def _load_onehot_feature(self, feature_data, row_norm=True):
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
        feature_data : (file_name, separator, column_keys)
            or List((file_name, separator, column_keys)) if there are multiple files
            file_name :  Which file to parse
            separator :  Separator in csv
            column_keys: How to parse each column in csv
                column_keys is a List, with following format [(key,type),(key,type),(key,type)...]
                if (key, type) is (None, None): 
                    the csv donot have column name, the whole csv will be loaded, 
                    and users can use (None, None) to tell the column should select and 
                    None to tell the column should be ignored. And the corresponding graph 
                    is treated as homograph.
                    The first not None column is treated as node_id
                else:
                    We will treat the graph as hetero. The first (key, type) is treated as node_id

        Return
        ------
        
        """
        if isinstance(feature_data, list):
            for d in feature_data:
                self._load_raw_label(d)
        else:
            file_path, separator, column_keys = feature_data
            assert isinstance(column_keys, list)
            if isinstance(column_keys[0], int):
                assert len(column_keys) == 2
                # homo graph
                info = pd.read_csv(file_path, sep=separator, header=None)
                node_info = info.iloc[:,column_keys[0]]
                start, end = column_keys[1]
                feature_info = info.iloc[:,start:end]

                id_map = self._id_maps['homo']
                node_ids = []
                for nid in node_info.to_numpy():
                    id = id_map[nid]
                    node_ids.append(id)

                node_ids = np.asarray(node_ids)
                features = feature_info.to_numpy(dtype=np.float32)
                features = sp.csr_matrix(features, dtype=np.float32)
                if row_norm:
                    features = row_normalize(features)
                features = np.array(features.todense())

                # sort features and node_ids
                features = features[node_ids]
                node_ids = np.arange(node_ids.shape[0])
                self._features.append(BasicFeature(node_ids, features))
            else:
                names = [key for (key, _) in column_keys]
                dtypes = {key : dtype for (key, dtype) in column_keys}

                info = pd.read_csv(file_path, sep=separator, header=None,
                                   names=names,
                                   dtype=dtypes)
                node_info = info.iloc[:,0]
                feature_info = info.iloc[:,1:]
                
                id_map = self._id_maps[names[0]]
                assert id_map is not None
                node_ids = []
                for nid in node_info.to_numpy():
                    id = id_map[nid]
                    node_ids.append(id)

                node_ids = np.asarray(node_ids)
                features = feature_info.to_numpy()
                features = sp.csr_matrix(features, dtype=np.float32)
                if row_norm:
                    features = row_normalize(features)
                features = np.array(features.todense())

                # sort features and node_ids
                features = features[node_ids]
                node_ids = np.arange(node_ids.shape[0])
                self._features.append(BasicFeature(node_ids, features, is_homo=False, node_type=names[0]))

    def _load_raw_label(self, label_data):
        r"""parse label data

        Parameters
        ----------
        label_data : (file_name, separator, column_keys)
            or List((file_name, separator, column_keys)) if there are multiple files
            file_name :  Which file to parse
            separator :  Separator in csv
            column_keys: How to parse each column in csv
                column_keys is a List, with following format [(key,type),(key,type),(key,type)...]
                or [idx, idx]
                if column_keys in format [idx, idx]
                    The csv donot have column name, it is parsed according to column idx. 
                    The should exist only two idxes, first is treated as src, second is treated as label.
                else:
                    Two (key, type) pairs should be provided here. Fist is treated as src, 
                    Second is treated as Label.

        Return
        ------
        labels : List
            A List of BasicLabel
        """
        if isinstance(label_data, list):
            for d in label_data:
                self._load_raw_label(d)
        else:
            file_path, separator, column_keys = label_data
            assert isinstance(column_keys, list)
            if isinstance(column_keys[0], int):
                assert len(column_keys) == 2
                # homo graph
                info = pd.read_csv(file_path, sep=separator, header=None)
                info = info.iloc[:,column_keys]

                # now parse label in (id, value) pairs
                pairs = []
                label_map = {}
                id_map = self._id_maps['homo']
                for _, row_val in info.iterrows():
                    src, label = row_val
                    src_id = id_map[src]
                    label_id = get_id(label_map, label)
                    pairs.append((src_id, label_id))
                
                pairs = np.asarray(pairs)
                ids = pairs[:,0]
                labels = pairs[:, 1]
                self._labels.append((BasicLabel((ids, labels), id_map, label_map)))
                self._label_map = label_map
            else:
                names = [key for (key, dtype) in column_keys]
                dtypes = {key : dtype for (key, dtype) in column_keys}
                # hetero graph
                assert len(column_keys) == 2, "Each label should in format of src: label"

                info = pd.read_csv(file_path, sep=separator, header=None,
                                   names=names,
                                   dtype=dtypes)

                # now parse label in (id, value) pairs
                pairs = []
                label_map = {}
                id_map = self._id_maps[names[0]]
                assert id_map is not None

                for _, row_val in info.iterrows():
                    src, label = row_val
                    src_id = id_map[src]
                    label_id = get_id(label_map, label)
                    pairs.append(src_id, label_id)
                pairs = np.asarray(pairs)
                ids = pairs[:,0]
                labels = pairs[:, 1]
                self._labels.append((BasicLabel((ids, labels),
                                                     id_map,
                                                     label_map,
                                                     False,
                                                     name[0],
                                                     name[1])))
                self._label_map = label_map

    def _build_graph(self, self_loop=True):
        if len(self._triplets) == 1:
            raw_graph = self._triplets[0]
            edges = raw_graph.edges
            adj = sp.coo_matrix((np.ones(edges.shape[0]),
                                (edges[:, 0], edges[:, 1])),
                                shape=(raw_graph.src_range, raw_graph.dst_range),
                                dtype=np.float32)
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
            ids = F.tensor(ids, dtype=np.int32).to(device)
            labels = F.tensor(labels, dtype=np.int64).to(device)
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