import os
from collections import OrderedDict
import rdflib as rdf
import re

import networkx as nx
import numpy as np

import dgl
import dgl.backend as F
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url

class RDFNodeEntity:
    """Class for entities
    Parameters
    ----------
    obj : str
        obj URI
    n_type : str
        Type of this entity
    """
    def __init__(self, obj, n_type):
        self.obj = obj
        self.n_type = n_type

    def __str__(self):
        return '{}|{}'.format(self.n_type, self.obj)

class RDFFeatureEntity:
    """Class for entities
    Parameters
    ----------
    value : str
        feature value
    n_type : str
        Type of this entity
    """
    def __init__(self, value, n_type):
        self.value = value
        self.n_type = n_type

    def __str__(self):
        return '{}|{}'.format(self.n_type, self.value)

class RDFRelation:
    """Class for relations
    Parameters
    ----------
    obj : str
        obj URI
    cls : str
        Type of this relation
    """
    def __init__(self, obj, r_type):
        self.obj = obj
        self.r_type = r_type

    def __str__(self):
        return '{}|{}'.format(self.r_type, self.obj)

def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

class RDFLoader():
    def __init__(self):
        pass

    def load_raw_tuples(self, data_path):
        raw_rdf_graphs = []
        for i, filename in enumerate(os.listdir(data_path)):
            fmt = None
            if filename.endswith('nt'):
                fmt = 'nt'
            elif filename.endswith('n3'):
                fmt = 'n3'
            elif filename.endswith('ttl'):
                fmt = 'ttl'
            elif filename.endswith('xml'):
                fmt = 'xml'
            if fmt is None:
                continue
            g = rdf.Graph()
            print('Parsing file %s ...' % filename)
            print(os.path.join(data_path, filename))
            g.parse(os.path.join(data_path, filename), format=fmt)
            raw_rdf_graphs.append(g)
        return raw_rdf_graphs

    def process_raw_tuples(self, raw_tuples):
        pass

class MUTAGDataset(RDFLoader):
    """MUTAG dataset.

    insert_reverse : bool, optional
        If true, add reverse edge and reverse relations to the final graph.
    """

    d_entity = re.compile("d[0-9]")
    bond_entity = re.compile("bond[0-9]")

    is_mutagenic = rdf.term.URIRef("http://dl-learner.org/carcinogenesis#isMutagenic")
    in_bound = rdf.term.URIRef("http://dl-learner.org/carcinogenesis#inBond")
    has_bound = rdf.term.URIRef("http://dl-learner.org/carcinogenesis#hasBond")
    rdf_type = rdf.term.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    rdf_subclassof = rdf.term.URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    rdf_domain = rdf.term.URIRef("http://www.w3.org/2000/01/rdf-schema#domain")

    entity_prefix = 'http://dl-learner.org/carcinogenesis#'
    relation_prefix = entity_prefix

    def __init__(self, insert_reverse=True):
        url='https://raw.githubusercontent.com/SmartDataAnalytics/DL-Learner/develop/examples/carcinogenesis/carcinogenesis.owl'
        download_dir = get_download_dir()
        name = "carcinogenesis.xml"
        file_path = os.path.join(download_dir, name)
        download(url, path=file_path)
        self._dir = download_dir
        self._insert_reverse = insert_reverse
        self.predict_category = "d"

        raw_rdf_graphs = self.load_raw_tuples(self._dir)
        self.process_raw_tuples(raw_rdf_graphs)
        print('#Training samples:', len(self.train_idx))
        print('#Testing samples:', len(self.test_idx))
        print('#Classes:', self.num_classes)
        print('Predict category:', self.predict_category)

    def parse_sbj(self, subject):
        if isinstance(subject, rdf.Literal):
            return RDFNodeEntity(str(term), "_Literal")
        elif isinstance(subject, rdf.BNode):
            return None
        entstr = str(subject)
        if entstr.startswith(self.entity_prefix):
            inst = entstr[len(self.entity_prefix):]
            if self.d_entity.match(inst):
                cls = 'd'
            elif self.bond_entity.match(inst):
                cls = 'bond'
            else:
                cls = None
            return RDFNodeEntity(entstr, cls)
        else:
            return None

    def parse_pred(self, predication):
        if predication == self.is_mutagenic:
            return None
        relstr = str(predication)
        if relstr.startswith(self.relation_prefix):
            cls = relstr[len(self.relation_prefix):]
            return RDFRelation(relstr, cls)
        else:
            cls = relstr.split('/')[-1]
            return RDFRelation(relstr, cls)

    def parse_obj(self, obj):
        if isinstance(obj, rdf.Literal):
            return RDFNodeEntity(str(obj), "_Literal")
        elif isinstance(obj, rdf.BNode):
            return None
        entstr = str(obj)
        if entstr.startswith(self.entity_prefix):
            inst = entstr[len(self.entity_prefix):]
            if self.d_entity.match(inst):
                cls = 'd'
            elif self.bond_entity.match(inst):
                cls = 'bond'
            else:
                cls = None
            return RDFNodeEntity(entstr, cls)
        else:
            return None

    def process_tuples(self, sbj, rel, obj):
        if sbj is None or rel is None or obj is None:
            return None

        if not rel.obj.startswith('http://dl-learner.org/carcinogenesis#'):
            obj.n_type = 'SCHEMA'
            if sbj.n_type is None:
                sbj.n_type = 'SCHEMA'
        if obj.n_type is None:
            obj.n_type = rel.r_type

        assert sbj.n_type is not None and obj.n_type is not None
        
        return (sbj, rel, obj)

    def process_raw_tuples(self, raw_rdf_graphs):
        triplets = {}
        mg = nx.MultiDiGraph()
        ent_classes = OrderedDict()
        rel_classes = OrderedDict()
        entities = OrderedDict()
        labels = OrderedDict()
        dataset_pairs = []

        src = []
        dst = []
        ntid = []
        etid = []

        mutag_graph = raw_rdf_graphs[0]
        for (sbj, pred, obj) in mutag_graph:
            if pred in triplets:
                triplets[pred].append((sbj, pred, obj))
            else:
                triplets[pred] = []
                triplets[pred].append((sbj, pred, obj))

        for key, triples in triplets.items():
            if key == self.is_mutagenic:
                continue
            for (sbj, pred, obj) in triples:
                sbjent = self.parse_sbj(sbj)
                rel = self.parse_pred(pred)
                objent = self.parse_obj(obj)

                processed = self.process_tuples(sbjent, rel, objent)
                if processed is None:
                    # ignored
                    continue

                sbjclsid = _get_id(ent_classes, sbjent.n_type)
                objclsid = _get_id(ent_classes, objent.n_type)
                relclsid = _get_id(rel_classes, rel.r_type)
                mg.add_edge(sbjent.n_type, objent.n_type, key=rel.r_type)
                if self._insert_reverse:
                    mg.add_edge(objent.n_type, sbjent.n_type, key='rev-%s' % rel.r_type)
                # instance graph
                src_id = _get_id(entities, str(sbjent))
                if len(entities) > len(ntid):  # found new entity
                    ntid.append(sbjclsid)
                dst_id = _get_id(entities, str(objent))
                if len(entities) > len(ntid):  # found new entity
                    ntid.append(objclsid)
                src.append(src_id)
                dst.append(dst_id)
                etid.append(relclsid)

        # handle label
        is_mutagenic_triplets = triplets[self.is_mutagenic]
        for (sbj, pred, obj) in is_mutagenic_triplets:
            print(sbj)
            sbj_id = _get_id(entities, str(sbj))
            label = _get_id(labels, str(obj))
            dataset_pairs.append((sbj_id, label))

        src = np.array(src)
        dst = np.array(dst)
        ntid = np.array(ntid)
        etid = np.array(etid)
        ntypes = list(ent_classes.keys())
        etypes = list(rel_classes.keys())

        # add reverse edge with reverse relation
        if self._insert_reverse:
            print('Adding reverse edges ...')
            newsrc = np.hstack([src, dst])
            newdst = np.hstack([dst, src])
            src = newsrc
            dst = newdst
            etid = np.hstack([etid, etid + len(etypes)])
            etypes.extend(['rev-%s' % t for t in etypes])

        self.build_graph(mg, src, dst, ntid, etid, ntypes, etypes)
        self.split_dataset(dataset_pairs, labels)

    def split_dataset(self, dataset_pairs, label_dict):
        total = len(dataset_pairs)
        train_set_size = int(total * 0.9)
        entities, labels = zip(*dataset_pairs)
        train_entities = entities[:train_set_size]
        test_entities = entities[train_set_size:]

        self.train_idx = F.tensor(train_entities)
        self.test_idx = F.tensor(test_entities)
        self.labels = F.tensor(labels).long()
        self.num_classes = len(label_dict)


    def build_graph(self, mg, src, dst, ntid, etid, ntypes, etypes):
        # create homo graph
        print('Creating one whole graph ...')
        g = dgl.graph((src, dst))
        g.ndata[dgl.NTYPE] = F.tensor(ntid)
        g.edata[dgl.ETYPE] = F.tensor(etid)
        print('Total #nodes:', g.number_of_nodes())
        print('Total #edges:', g.number_of_edges())

        # convert to heterograph
        print('Convert to heterograph ...')
        hg = dgl.to_hetero(g,
                           ntypes,
                           etypes,
                           metagraph=mg)
        print('#Node types:', len(hg.ntypes))
        print('#Canonical edge types:', len(hg.etypes))
        print('#Unique edge type names:', len(set(hg.etypes)))
        self.graph = hg

