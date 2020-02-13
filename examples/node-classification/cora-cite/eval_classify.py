"""GCN example, copy from https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn
"""
import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

from gcn import GCN
from cora_loader import CoraDataset

def evaluate(model, features, test_set):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[test_set[0]]
        labels = test_set[1]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    if args.gpu > 0:
        cuda = True
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
        cuda = False
    cora_data = CoraDataset(device, valid_ratio=0.1, test_ratio=0.2)
    features = cora_data.features
    test_set = cora_data.test_set
    g = cora_data.g

    in_feats = features['homo'].shape[1]
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                cora_data.n_class,
                args.n_layers,
                F.relu)
    model.load_state_dict(torch.load(args.model_path))
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    print()
    acc = evaluate(model, features['homo'], test_set)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--model_path", type=str,
            help='save path for model')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
