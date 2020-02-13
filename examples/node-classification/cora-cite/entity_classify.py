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
    valid_set = cora_data.valid_set
    train_set = cora_data.train_set
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
                F.relu,
                args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features['homo'])
        loss = loss_fcn(logits[train_set[0]], train_set[1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features['homo'], valid_set)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features['homo'], test_set)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
