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


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
OUTPUT_DIR_PATH = ROOT_DIR + '/outputs'
INPUTS_DIR_PATH = ROOT_DIR + '/inputs'
ROOT_PAN24_PATH = ROOT_DIR + '/datasets/pan24'
ROOT_SEMEVAL24_PATH = ROOT_DIR + '/datasets/semeval24'


def save_data(data, file_name, path=OUTPUT_DIR_PATH, format_file='.pkl', compress=False):
    path_file = os.path.join(path, file_name + format_file)
    joblib.dump(data, path_file, compress=compress)


def load_data(file_name, path=OUTPUT_DIR_PATH, format_file='.pkl', compress=False):
    path_file = os.path.join(path, file_name + format_file)
    return joblib.load(path_file)


def read_json(file_path):
  print(file_path)
  df = pd.read_json(file_path, lines=True)
  df = df.sort_values('id', ascending=True)
  return df


def save_json(data, file_path):
    with open(file_path, "w") as outfile:
        for element in data:  
            json.dump(element, outfile)  
            outfile.write("\n")  


def plot_chart(x, y, labels, width=0.5, figsize=(20,4)):
  plt.subplots(figsize=figsize)
  plt.xticks(rotation='vertical')
  plt.bar(x, y, tick_label = labels,width = width, color = ['red', 'green'])
  plt.show()


def build_pan_test_file(test_set):
    #{"id": "iixcWBmKWQqLAwVXxXGBGg", "text1": "...", "text2": "..."}
    test_set = shuffle(test_set)
    pan_test_human = shuffle(test_set[test_set['class'] == 1])
    pan_test_machine = shuffle(test_set[test_set['class'] == 0])
    pan24_test = []
    rand_opt = [0,1]
    for i in range(0, min(len(pan_test_human), len(pan_test_machine))):
        texts = [pan_test_human.iloc[i]['text'], pan_test_machine.iloc[i]['text']]
        labels = [pan_test_human.iloc[i]['class'], pan_test_machine.iloc[i]['class']]
        rand = random.choice(rand_opt)
        d = {
            'id': i,
            'label1': int(labels[rand]),
            'label2': int(labels[rand-1]),
            'text1': texts[rand],
            'text2': texts[rand-1],
        }
        pan24_test.append(d)
    return pan24_test

def data_augmentation():
    source = 'wikipedia'
    subtaskA_train_monolingual = read_json(file_path=ROOT_SEMEVAL24_PATH + '/subtaskA_train_monolingual.jsonl')
    #print(subtaskA_train_monolingual.info())
    #print("\n", semeval_human_set['source'].value_counts())
    semeval_human_set = subtaskA_train_monolingual[(subtaskA_train_monolingual['label'] == 0)]
    semeval_human_subset = semeval_human_set[(semeval_human_set['source'] == source)]
    semeval_human_subset.rename(columns={'label': 'class'}, inplace=True)
    semeval_human_subset['id'] = semeval_human_subset.apply(lambda row: f'semeval-{source}-{str(row["id"])}', axis=1)
    semeval_human_subset['class'] = 1
    del semeval_human_subset['source']
    return semeval_human_subset


def dataset_partition(dataset):
    pan24_machine = dataset[dataset['class'] == 0]
    pan24_human = dataset.loc[dataset['class'] == 1]
    train_machine, test_machine = train_test_split(pan24_machine, test_size=0.2)
    train_human, test_human = train_test_split(pan24_human, test_size=0.2)
    train = pd.concat([train_human, train_machine])
    test = pd.concat([test_human, test_machine])
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    del train['level_0']
    del test['level_0'] 
    return train, test


def build_pipeline(model):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1))),
        ('clf', CalibratedClassifierCV(model())),
    ])
    return pipeline


def main(known_args, unknown_args):
    #print(known_args)
    print(unknown_args)
    model = 'SGDClassifier'

    '''
    if unknown_args[0] == 'proccess_data':
        # read and process data
        data_human_df = read_json(ROOT_PAN24_PATH + '/human.jsonl')
        data_human_df['class'] = 1 # human
        data_machines_df = pd.concat([read_json(f) for f in glob.glob(ROOT_PAN24_PATH + '/machines/*.jsonl')])
        data_machines_df.reset_index(inplace=True)
        del data_machines_df['index']
        data_machines_df['class'] = 0 # machine
        dataset = pd.concat([data_human_df, data_machines_df])
        print(dataset.info())
        print(dataset['class'].value_counts())

        # data aufmentation for human data, using semeval2024
        semeval_human_subset = data_augmentation() 
        pan24_dataset = dataset.copy(deep=True)
        pan24_dataset['model'] = None
        for index, row in pan24_dataset.iterrows():
            if row['class'] == 0:
                split_id = row['id'].split('/')
                pan24_dataset.loc[index,'model'] = split_id[0]

        pan24_machine = pan24_dataset[pan24_dataset['class'] == 0]
        pan24_human = pan24_dataset.loc[pan24_dataset['class'] == 1]
        pan24_human['model'] = 'human'
        pan24_human = pd.concat([semeval_human_subset, pan24_human]) # concat semeval data
        pan24_dataset = pd.concat([pan24_human, pan24_machine])
        pan24_dataset.reset_index(inplace=True)
        pan24_dataset = shuffle(pan24_dataset)

        print(pan24_dataset.info())
        print(pan24_dataset['class'].value_counts())

        # data partition: train and test
        train_set, test_set = dataset_partition(pan24_dataset)
        save_data(data=train_set, file_name='train_set')
        save_data(data=test_set, file_name='test_set')
        test_set_dict = build_pan_test_file(test_set)
        save_json(data=test_set_dict, file_path=INPUTS_DIR_PATH + "/test.jsonl")

    if unknown_args[0] == 'train':
        train_set = load_data(file_name='train_set')
        test_set = read_json(file_path=INPUTS_DIR_PATH + '/test.jsonl')
        #test_set = load_data(file_name='test_set')

        print(train_set.info())
        print(test_set.info())

        # build baseline model
        clf_models = {
            'LinearSVC': LinearSVC,
            'MultinomialNB': MultinomialNB,
            'LogisticRegression': LogisticRegression,
            'SGDClassifier': SGDClassifier
        }
        
        print('trainin model...')
        pipeline = build_pipeline(model = clf_models[model])
        # train model
        pipeline.fit(train_set['text'], train_set['class'])
        save_data(data=pipeline, file_name='clf_train_'+model)
        print('Done!')
    else:
        pipeline = load_data(file_name='clf_train_' + model)
        train_set = load_data(file_name='train_set')
        test_set = read_json(file_path=INPUTS_DIR_PATH + '/test.jsonl')
    '''

    # predictions
    pipeline = load_data(file_name='clf_train_' + model)
    train_set = load_data(file_name='train_set')
    
    test_set = read_json(file_path=unknown_args[0])
    print(test_set.info())

    #test_set = read_json(file_path=INPUTS_DIR_PATH + '/test.jsonl')
    test_set = test_set.to_dict('records')
    test_set = test_set[:]
    #print(test_set[0])

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

    #save_json(data=predictions, file_path=OUTPUT_DIR_PATH + "/preds.jsonl")
    save_json(data=predictions, file_path=unknown_args[1] + "/preds.jsonl")

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