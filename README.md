# Twitter sentiment classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marekb-sci/twitter_sentiment_classification/blob/main/colab_demo.ipynb)

This repo implements model for twitt sentiment classification (http://help.sentiment140.com/for-students)

1. Installation
    1. install requirements
    1. testing
1. Usage
    1. Prepare data
    1. Training
    1. Inference
1. TODO list

Colab 

## Installation
Install required packages and test scripts

Clone repository and walk into working directory


```bash
git clone https://github.com/marekb-sci/twitter_sentiment_classification.git
```

```bash
cd twitter_sentiment_classification
```

Install requirements


```bash
pip install -r requirements.txt
```


Perform testing: (optional, partially implemented)


```python
pytest tests/
```
## Usage

### Prepare data

Download:


```bash
wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip -d data

# alternative in Colab:
# !gdown '0B04GJPshIjmPRnZManQwWEdTZjg'
# !unzip /content/trainingandtestdata.zip -d /content/data

```
    

Preprocess data:


```bash
python prepare_train_data.py \
--input-path data/training.1600000.processed.noemoticon.csv \
--output-dir data
```

### Training


```bash
python train.py \
--train-data data/training.1600000.processed.noemoticon_train.csv \
--val-data data/training.1600000.processed.noemoticon_val.csv \
--epochs 3
```

### Inference

Write data for inference in text file. Each twitt should be in a separate line. Sample file with data ready for inference is located at `sample_data/sample_for_inference.txt`


```bash
cat sample_data/sample_for_inference.txt
```

The ouptput should be:
```
@RobCairns Thanx..I improved once, so I'm hopinng I will do so again..more function is still better than less. Thanx for understanding.
OK, I'm done ppl, I will not reply to anymore bball digs...its taken up my entire day so far! Let me get back to fashion and beauty
@imjstsayin I'm enjoying the sunshine! Maybe a beach day today  How are you?
```
Run inference:

```bash
python inference.py \
--model-dir output_model \
--data sample_data/sample_for_inference.txt \
--output-file sample_data_infernece.txt
```

Check results:
```bash
cat sample_data_infernece.txt
```
The ouptput will look like:
```
LABEL_1 0.7134
LABEL_0 0.7452
LABEL_1 0.8012
```
where first column indicates the predicted label and the second column: score of this label

## TODO

- test accuracy for different settigs
- make inference labels readable  
- make tests

