
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import random
import glob
import math 

import utils

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
        model = [pan_test_human.iloc[i]['model'], pan_test_machine.iloc[i]['model']]
        rand = random.choice(rand_opt)
        d = {
            'id': i,
            'label1': int(labels[rand]),
            'label2': int(labels[rand-1]),
            'model1': model[rand],
            'model2': model[rand-1],
            'text1': texts[rand],
            'text2': texts[rand-1],
        }
        pan24_test.append(d)
    return pan24_test

def data_augmentation():
    source = 'wikipedia'
    subtaskA_train_monolingual = utils.read_json(file_path=utils.ROOT_SEMEVAL24_PATH + '/subtaskA_train_monolingual.jsonl')
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


def main():
    #***** 1) read and process data
    data_human_df = utils.read_json(utils.ROOT_PAN24_PATH + '/human.jsonl')
    data_human_df['class'] = 1 # human
    data_machines_df = pd.concat([utils.read_json(f) for f in glob.glob(utils.ROOT_PAN24_PATH + '/machines/*.jsonl')])
    data_machines_df.reset_index(inplace=True)
    del data_machines_df['index']
    data_machines_df['class'] = 0 # machine
    dataset = pd.concat([data_human_df, data_machines_df])
    print(dataset.info())
    print(dataset['class'].value_counts())

    #***** 2) extract info from id: class, topic, model, art-id
    pan24_dataset = dataset.copy(deep=True)
    pan24_dataset['model'] = None
    for index, row in pan24_dataset.iterrows():
        if row['class'] == 0:
            split_id = row['id'].split('/')
            pan24_dataset.loc[index,'topic'] = split_id[1].split('-')[-1]
            pan24_dataset.loc[index,'art-id'] = split_id[2]
            pan24_dataset.loc[index,'model'] = split_id[0]

    pan24_machine = pan24_dataset[pan24_dataset['class'] == 0]
    pan24_human = pan24_dataset.loc[pan24_dataset['class'] == 1]
    pan24_human['model'] = 'human'
    print(pan24_human.info())
    #print(pan24_human['model'].value_counts())
    #print(pan24_human['topic'].value_counts())
    
    #***** extact 1087 instances from Machine set (balance in models and topics)
    pan24_machine_model_partitions = pd.DataFrame()
    for model in pan24_machine['model'].unique():
        pan24_machine_topic_partitions = []
        machine_model = pan24_machine.loc[pan24_machine['model'] == model]
        for topic in pan24_machine['topic'].unique():
            machine_model_topic = machine_model.loc[machine_model['topic'] == topic]
            machine_model_topic = machine_model_topic[:math.floor(len(machine_model_topic)*0.087)]
            pan24_machine_topic_partitions.append(machine_model_topic)
        pan24_machine_topic_partitions_tmp = pd.concat(pan24_machine_topic_partitions)
        pan24_machine_model_partitions = pd.concat([pan24_machine_model_partitions, pan24_machine_topic_partitions_tmp])
    
    print(pan24_machine_model_partitions.info())
    #print(pan24_machine_model_partitions['model'].value_counts())
    #print(pan24_machine_model_partitions['topic'].value_counts())

    #***** concat Human and Machine sets
    pan24_dataset = pd.concat([pan24_human, pan24_machine_model_partitions])
    pan24_dataset.reset_index(inplace=True)
    pan24_dataset = shuffle(pan24_dataset)
    print(pan24_dataset.info())

    #***** data partition: train and test
    train_set, test_set = dataset_partition(pan24_dataset)
    utils.save_data(data=train_set, file_name='Partition_A_train_set')
    utils.save_data(data=test_set, file_name='Partition_A_test_set')
    
    #***** build pan test_file and test_truth_file format {id: 1, text1: "abc", text2: "def"}
    test_set_lst = build_pan_test_file(test_set)
    test_set_truth_lst = []
    for row in test_set_lst:
        is_human_txt = 0
        if row['label2'] == 1:
            is_human_txt = 1
        d = {'id': row['id'], 'is_human': is_human_txt}
        test_set_truth_lst.append(d)

    train_set = train_set[['index', 'id', 'topic', 'class', 'model', 'art-id', 'text']]
    train_set = train_set.to_dict('records')

    utils.save_json(data=train_set, file_path=utils.INPUTS_DIR_PATH + "/Partition_A_train_set.jsonl")
    utils.save_json(data=test_set_lst, file_path=utils.INPUTS_DIR_PATH + "/Partition_A_test_set.jsonl")
    utils.save_json(data=test_set_truth_lst, file_path=utils.INPUTS_DIR_PATH + "/Partition_A_test_truth.jsonl")


if __name__ == '__main__':
    main()