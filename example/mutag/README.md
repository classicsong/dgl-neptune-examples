# RGCN node classification example with RDF

This example shows how to use dgl to train data stored in RDF format and generate Sparql cmds. We can use Neptune to output data in RDF format and use code here to do RGCN model training. With trained model, we can infer type of new nodes and generate INSERT cmds in Sparql.

## Dataset
The MUTAG dataset is distributed as an example dataset for the DL-Learner toolkit (http://dl-learner.org). It contains information about complex molecules that are potentially carcinogenic, which is given by the \texttt{isMutagenic} property.

## Task Specification
The MUTAG data is stored as RDF in xml format which can be loaded by Neptune directly. We will build a DGL hetero-graph using MUTAG dataset and train a RGCN model to predict whether a molecules is carcinogenic. In this example we take 80% molecules (<http://dl-learner.org/carcinogenesis#dxxx>) of data as training set and take 20% molecules as nodes that we want to predict their type (True or False). Finally, we will generate Sparql for these 20% molecules.

## How to use
### Train
> python3 entity_classification.py --l2norm 5e-4 --n-bases 30 --model_path="tmp.pt“


### Eval
The CMD will generate Sparql CMD.
> python3 eval_classification.py --n-bases 30 --model_path "tmp.pt"

### Issue neptune insert
> curl -X POST --data-binary \'$CMD' http://xxx.neptune.amazonaws.com:8182/sparql
