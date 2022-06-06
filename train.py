import argparse, os
import json
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


from datasets import (
    Dataset,
    load_metric,
    )

from transformers import (
    pipeline,
    AutoTokenizer, 
    DataCollatorWithPadding, 
    AutoModelForSequenceClassification,
    )

def create_dataset(csv_filename, labels=None):
    df = pd.read_csv(csv_filename)
    assert all([x in df.columns for x in ['label', 'text']])
    if labels is not None:
        df = df.loc[df['label'].isin(labels)]
    ds = Dataset.from_pandas(df[['label', 'text']])
    return ds

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # training parameters.
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=64)

    # Data, model, and output directories
    parser.add_argument('--train-data', type=str, help='path to train data csv file with columns: "text", "label"')
    parser.add_argument('--val-data', type=str, help='path to validation data csv file with columns: "text", "label"')
    parser.add_argument('--training-output', type=str, default='training_output')
    parser.add_argument('--output-model', type=str, default='output_model')

    args = parser.parse_args()

    # load datasets
    ds_train_raw = create_dataset(args.train_data)
    labels = np.unique(ds_train_raw['label'])
    n_classes = len(labels)
    ds_val_raw = create_dataset(args.val_data, labels=labels)

    # tokenization
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased') 
    ds_train = ds_train_raw.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
    ds_val = ds_val_raw.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=n_classes)

    training_args = TrainingArguments(
        output_dir=args.training_output,
        learning_rate=2e-5,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=ds_train,
        eval_dataset=ds_val,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=ds_val)

    print("***** Eval results *****")
    for key, value in sorted(eval_result.items()):
        print(f"{key} = {value}\n")

    #print and save evaluation resluts
    with open(os.path.join(args.training_output, "eval_results.json"), "w") as f:
        json.dump(eval_result, f)

    # save pipeline
    # model_pipeline = pipeline(task='text-classification', model=model, tokenizer=tokenizer)
    # model_pipeline.save_pretrained(args.output_model)
    trainer.save_model(args.output_model)



#     import random, sys, argparse, os, logging, torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from datasets import load_from_disk

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()

#     # hyperparameters sent by the client are passed as command-line arguments to the script.
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--train-batch-size", type=int, default=32)
#     parser.add_argument("--eval-batch-size", type=int, default=64)
#     parser.add_argument("--save-strategy", type=str, default='no')
#     parser.add_argument("--save-steps", type=int, default=500)
#     parser.add_argument("--model-name", type=str)
#     parser.add_argument("--learning-rate", type=str, default=5e-5)

#     # Data, model, and output directories
#     parser.add_argument("--output-dir", type=str, default="training_output")
#     parser.add_argument("--model-dir", type=str, default="output_model")

#     args, _ = parser.parse_known_args()

#     # load datasets
#     train_dataset = load_from_disk(args.train_dir)
#     valid_dataset = load_from_disk(args.valid_dir)
    
#     logger = logging.getLogger(__name__)
#     logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
#     logger.info(f" loaded valid_dataset length is: {len(valid_dataset)}")

#     # compute metrics function for binary classification
#     def compute_metrics(pred):
#         labels = pred.label_ids
#         preds = pred.predictions.argmax(-1)
#         precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
#         acc = accuracy_score(labels, preds)
#         return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

#     # download model from model hub
#     model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    
#     # download the tokenizer too, which will be saved in the model artifact and used at prediction time
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)

#     # define training args
#     training_args = TrainingArguments(
#         output_dir=args.model_dir,
#         num_train_epochs=args.epochs,
#         per_device_train_batch_size=args.train_batch_size,
#         per_device_eval_batch_size=args.eval_batch_size,
#         save_strategy=args.save_strategy,
#         save_steps=args.save_steps,
#         evaluation_strategy="epoch",
#         logging_dir=f"{args.output_data_dir}/logs",
#         learning_rate=float(args.learning_rate),
#     )

#     # create Trainer instance
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#     )




