# GCN node classification example with CSV as input
This example shows how to use dgl to train data stored in CSV format and generate Gremlin cmds. We can use Neptune to output data using neptune-export tool (https://github.com/awslabs/amazon-neptune-tools/tree/master/neptune-export) to export data in CSV format and use code here to do GCN model training. With trained model, we can infer type of new nodes and generate INSERT cmds in Gremlin.

## Dataset
The Cora dataset (https://relational.fit.cvut.cz/dataset/CORA) consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.

## Task Specification
The Cora dataset is stored as CSV-like format which can be loaded through Neptune Gremlin interface.

## How to use
### Upload data into Neptune
We have prepare the data for you, cora.cites.csv for citation relationships and cora.content.csv for paper feature specification.
You can store data in S3 and let Neptune load it from S3.

```
aws s3 cp cora.cites.csv s3://<YOUR_S3_BUCKET>/cora/cora.cites.csv
aws s3 cp cora.content.csv s3://<YOUR_S3_BUCKET>/cora/cora.content.csv

# upload to csv
curl -X POST -H 'Content-Type: application/json' https://<YOUR_NEPTUNE_ADDR>.neptune.amazonaws.com:8182/loader -d '{"source" : "https://<YOUR_S3_BUCKET_ADDR>/cora/cora.cites.csv", "format" : "csv", "iamRoleArn" : "<YOUR_ARN>", "region" : "<YOUR_REGION>", "failOnError" : "FALSE", "parallelism" : "MEDIUM", "updateSingleCardinalityProperties" : "FALSE" }'
curl -X POST -H 'Content-Type: application/json' https://<YOUR_NEPTUNE_ADDR>.neptune.amazonaws.com:8182/loader -d '{"source" : "https://<YOUR_S3_BUCKET_ADDR>/cora/cora.content.csv", "format" : "csv", "iamRoleArn" : "<YOUR_ARN>", "region" : "<YOUR_REGION>", "failOnError" : "FALSE", "parallelism" : "MEDIUM", "updateSingleCardinalityProperties" : "FALSE" }'
```

### Export data using neptune-export
We use neptune-export to export data in neptune to CSV. For more details please refer to https://github.com/awslabs/amazon-neptune-tools/tree/master/neptune-export.

```
sudo apt install maven
mvn clean install
git clone https://github.com/awslabs/amazon-neptune-tools.git
cd neptune-export
bin/neptune-export.sh export-pg -d ~/data/cora -e <YOUR_NEPTUNE_ADDR>.neptune.amazonaws.com --format csv --batch-size 4 --max-content-length 16777216
```

You will see config.json edges/ and nodes/ three file/dirs in ~/data/cora/<timeline>/ directory. Now we can train the node classification model using data from ~/data/cora/<timeline>/edges and ~/data/cora/<timeline>/nodes.

### Train
You can directly run the training script to train data from cora dataset:
```
python3 entity_classify.py --gpu 0 --model_path cora.pt
```

The saved model is cora.pt

### Eval
You can directly run the testing script to test data from cora dataset:
```
python3 eval_classify.py --gpu 0 --model_path cora.pt
```

### Inference
You can directly run the inference script to infer test data from cora dataset and generate gremlin cmds
```
python3 infer_classify.py --gpu 0 --model_path cora.pt
```

The output is a list of following strings:
```
{"gremlin":"g.V(\"143476\").property(\"category\", \"Probabilistic_Methods\")"}
{"gremlin":"g.V(\"1116347\").property(\"category\", \"Reinforcement_Learning\")"}
```

You can using curl cmd to update gremlin database
```
curl -X POST -d '<CMD_FROM_INFER>' https://<YOUR_NEPTUNE_ADDR>.neptune.amazonaws.com:8182/gremlin
```

For example
```
curl -X POST -d '{"gremlin":"g.V(\"1116347\").property(\"category\", \"Reinforcement_Learning\")"}' https://<YOUR_NEPTUNE_ADDR>.neptune.amazonaws.com:8182/gremlin
```