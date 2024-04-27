import pandas as pd
import numpy as np
import random
import json
import random
import glob
import pprint
import os
import joblib
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics

from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate


import utils

metric = evaluate.load("accuracy")


def llm_baseline():
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    dataset = load_dataset("mteb/tweet_sentiment_extraction")
    df = pd.DataFrame(dataset['train'])
    print(df.info())
    return

    # Loading the dataset to train our model
    dataset = load_dataset("mteb/tweet_sentiment_extraction")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        #evaluation_strategy="epoch",
        per_device_train_batch_size=1,  # Reduce batch size here
        per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
        gradient_accumulation_steps=4
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()




def build_pipeline(model):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1))),
        ('clf', CalibratedClassifierCV(model())),
    ])
    return pipeline


def main(known_args, unknown_args):
    llm_baseline()
    return

    #print(known_args)
    print(unknown_args)
    algo_ml = 'SGDClassifier' # SGDClassifier, LinearSVC, MultinomialNB, LogisticRegression
    model_name = 'Model_A_clf_train_' # Model_A_clf_train_, pan24_smoke_train_
    train_set_name = 'Partition_A_train_set' # Partition_A_train_set, pan24_generative_authorship_smoke_train
    test_set_name  = ''  #Partition_A_test.jsonl

    if unknown_args[0] == 'train':
        #***** read data
        #train_set = utils.load_data(file_name=train_set_name)
        train_set = utils.read_json(file_path=utils.INPUTS_DIR_PATH + train_set_name + '.jsonl')
        print(train_set.info())

        # use only for pan24_smoke_train
        if train_set_name == 'pan24_generative_authorship_smoke_train':
            train_set_tmp_txt1 = pd.DataFrame({'id': train_set['id'], 'text': train_set['text1'], 'class': train_set['label_txt1']})
            train_set_tmp_txt2 = pd.DataFrame({'id': train_set['id'], 'text': train_set['text2'], 'class': train_set['label_txt2']})
            train_set = pd.concat([train_set_tmp_txt1, train_set_tmp_txt2])
            print(train_set.info())

        #***** build baseline model
        clf_models = {
            'LinearSVC': LinearSVC,
            'MultinomialNB': MultinomialNB,
            'LogisticRegression': LogisticRegression,
            'SGDClassifier': SGDClassifier
        }
        
        #***** train model
        print('training model...')
        pipeline = build_pipeline(model = clf_models[algo_ml])
    
        pipeline.fit(train_set['text'], train_set['class'])
        utils.save_data(data=pipeline, file_name=model_name + algo_ml)
        print('Done!')

    else:
        # predictions
        pipeline = utils.load_data(file_name=model_name + algo_ml)
        test_set = utils.read_json(file_path=unknown_args[0])
        #test_set = utils.read_json(file_path=utils.INPUTS_DIR_PATH + '/test.jsonl')
        print(test_set.info())

        test_set = test_set.to_dict('records')
        test_set = test_set[:]
        predictions = []

        for row in test_set:
            txt1_pred = pipeline.predict([row['text1']])
            txt2_pred = pipeline.predict([row['text2']])
            res = None
            #print(20*'*', row['id'])
            #print('       Txt1  Tx2',)
            #print('label: ', row['label1'], '   ', row['label2'])
            #print('pred:  ', txt1_pred[0], '   ', txt2_pred[0])

            if txt1_pred == 1:
                #print("TEXT 1 is Human")
                pred_proba = pipeline.predict_proba([row['text1']])[0]
                max_pred = max(pred_proba[0], pred_proba[1])
                res = {"id": row['id'], "is_human": round(1-max_pred,2)}
            elif txt2_pred == 1:
                #print("TEXT 2 is Human")
                pred_proba = pipeline.predict_proba([row['text2']])[0]
                max_pred = max(pred_proba[0], pred_proba[1])
                res = {"id": row['id'], "is_human": round(max_pred,2)}
            else: # check??
                res = {"id": row['id'], "is_human": 0.5}

            predictions.append(res)

        #utils.save_json(data=predictions, file_path=utils.OUTPUT_DIR_PATH + '/' + model_name + "preds.jsonl")
        utils.save_json(data=predictions, file_path=unknown_args[1] + '/' + model_name + "preds.jsonl")

        #predicted = pipeline.predict(test_set['text'])
        #print('Accuracy:', np.mean(predicted == test_set['class']))  
        #print(metrics.classification_report(test_set['class'], predicted,target_names=['machine', 'human']))
        #print("Matriz Confusion: ")
        #metrics.confusion_matrix(test_set['class'], predicted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input_dataset", help="input test data ", default='', type=str)
    parser.add_argument("-out", "--onput_dir", help="output dir to save pred data", default='', type=str)
    parser.add_argument("-type", "--exec_type", help="train or test exection ", default='train', type=str)
    parser.add_argument("-model", "--model", help="model to train or test ", default='SGDClassifier', type=str)
    known_args, unknown_args = parser.parse_known_args()
    known_args = vars(known_args)
    main(known_args, unknown_args)