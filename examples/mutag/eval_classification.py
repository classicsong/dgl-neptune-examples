import argparse
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import dgl.function as fn

from mutag_loader import MUTAGDataset

from entity_classification import EntityClassify

def main(args):
    # load graph data
    dataset = MUTAGDataset()

    g = dataset.graph
    category = dataset.predict_category
    num_classes = dataset.num_classes
    test_idx = dataset.test_idx
    labels = dataset.labels
    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        test_idx = test_idx.cuda()
        labels = labels.cuda()

    # create model
    model = EntityClassify(g,
                           args.n_hidden,
                           num_classes,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           use_self_loop=args.use_self_loop)

    model.load_state_dict(th.load(args.model_path))
    if use_cuda:
        model.cuda()
    model.eval()
    logits = model.forward()[category_id]
    results = logits[test_idx].argmax(dim=1)
    dataset.gen_sparql(test_idx.cpu().detach().numpy(), results.cpu().detach().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--model_path", type=str,
            help='model path to load')
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)