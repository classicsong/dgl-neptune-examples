# RGCN node classification example with RDF as input
This example shows how to use dgl to train data stored in RDF format and generate Sparql cmds. We can use Neptune to output data in RDF format and use code here to do RGCN model training. With trained model, we can infer type of new nodes and generate INSERT cmds in Sparql.

## Dataset
The MUTAG dataset is distributed as an example dataset for the DL-Learner toolkit (http://dl-learner.org). It contains information about complex molecules that are potentially carcinogenic, which is given by the \textit{isMutagenic} property.

## Task Specification
The MUTAG data is stored as RDF in xml format which can be loaded by Neptune directly. We will build a DGL hetero-graph using MUTAG dataset and train a RGCN model to predict whether a carcinogenesis isMutagenic. The dataset containers 340 carcinogenesis. In this example we take 80% carcinogenesis (<http://dl-learner.org/carcinogenesis#dxxx>) as training set and take 20% carcinogenesis as nodes that we want to predict their type (True or False). Finally, we will generate Sparql for these 20% carcinogenesis.

mutag_example_show.jpeg shows a small part of this dataset. In DGL, we will build a dgl.heterograph based on the graph structure.

## How to use
### Train
You can use following scripts to run the training, it includes: 1) loading data in xml/rdf format; 2) parse and build DGL heterograph; 3) training the RGCN model and 4) save the model in tmp.pt. The tunable params include n-hidden, lr, n-bases, n-layers, n-epochs and l2norm.
> python3 entity_classification.py --l2norm 5e-4 --n-bases 30 --model_path="tmp.pt"

### Eval
You can use following scripts to run the evaluation to infer the type of carcinogenesis in test set and generate Sparql CMD.
> python3 eval_classification.py --n-bases 30 --model_path "tmp.pt"

### Issue neptune insert
> curl -X POST --data-binary \'$CMD' http://xxx.neptune.amazonaws.com:8182/sparql
