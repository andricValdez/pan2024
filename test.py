
# import json module
import json
import os
import math 
import glob
import pandas as pd

import utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
OUTPUT_DIR_PATH = ROOT_DIR + '/outputs'
INPUTS_DIR_PATH = ROOT_DIR + '/inputs'
ROOT_PAN24_PATH = ROOT_DIR + '/datasets/pan24'
ROOT_SEMEVAL24_PATH = ROOT_DIR + '/datasets/semeval24'


def test_save_jsonl_data():
    # list of dictionaries of employee data
    data = [
            {"id": 1, "text1": "abc", "text2": "def"},
            {"id": 2, "text1": "abc", "text2": "def"},
            {"id": 3, "text1": "abc", "text2": "def"},
            {"id": 4, "text1": "abc", "text2": "def"},
            {"id": 5, "text1": "abc", "text2": "def"}      
        ]
    # convert into json
    final = json.dumps(data)
    with open(INPUTS_DIR_PATH + "/test.jsonl", "w") as outfile:
        for element in data:  
            json.dump(element, outfile)  
            outfile.write("\n")  
    # display
    print(final)


def test_topics_distro():
    topics = {
        "covid19": 61,
        "kabulairportattack": 58,
        "colonialpipelinehack": 54,
        "kylerittenhousenotguilty": 54,
        "twitterbanstrump": 52,
        "tigerwoodsaccident": 52,
        "michiganhighschoolshooting": 52,
        "harryandmeghan": 51,
        "hurricaneida": 49,
        "stimuluscheck": 48,
        "cnnchriscuomo": 46,
        "evergreensuezcanal": 46,
        "facebookoutage": 46,
        "tombradysuperbowl": 45,
        "capitolriot": 45,
        "rustmoviesetshooting": 44,
        "bideninauguration": 43,
        "kamalaharrisvicepresident": 42,
        "trumpimpeachment": 41,
        "colinpowelldead": 40,
        "wyominggabbypetito": 39,
        "georgefloydderekchauvin": 35,
        "winterstormtexas": 29,
        "citibank500millionmistake": 15
    }

    total = 0
    total2 = 0
    for key, value in topics.items():
        total += value
        total2 += math.floor(value*0.087)
        print(math.floor(value*0.087))
    
    print(total)
    print(total2)


def test_pan24_generative_authorship_smoke_20240411_0_training():
    #***** read and process data
    data_human_df = utils.read_json(utils.ROOT_PAN24_PATH + '/human.jsonl')
    data_human_df['class'] = 1 # human
    data_machines_df = pd.concat([utils.read_json(f) for f in glob.glob(utils.ROOT_PAN24_PATH + '/machines/*.jsonl')])
    data_machines_df.reset_index(inplace=True)
    del data_machines_df['index']
    data_machines_df['class'] = 0 # machine
    dataset = pd.concat([data_human_df, data_machines_df])

    #***** extract info from id: class, topic, model, art-id
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
    pan24_dataset = pd.concat([pan24_human, pan24_machine])
    pan24_dataset.reset_index(inplace=True)

    #***** extract PAN dataset: pan24_generative_authorship_smoke_20240411_0_training
    pan24_generative_authorship_smoke_train = utils.read_json(file_path= utils.INPUTS_DIR_PATH + '/pan24-generative-authorship-smoke-20240411_0-training.jsonl')
    print(pan24_generative_authorship_smoke_train.info())

    #***** label class and model for pan24_generative_authorship_smoke_20240411_0_training
    for index, row in pan24_generative_authorship_smoke_train.iterrows():
        row_txt1 = pan24_dataset.loc[pan24_dataset['text'] == row['text1']]
        row_txt2 = pan24_dataset.loc[pan24_dataset['text'] == row['text2']]

        pan24_generative_authorship_smoke_train.loc[index,'label_txt1'] = row_txt1['class'].values[0]
        pan24_generative_authorship_smoke_train.loc[index,'label_txt2'] = row_txt2['class'].values[0]
        pan24_generative_authorship_smoke_train.loc[index,'model_txt1'] = row_txt1['model'].values[0]
        pan24_generative_authorship_smoke_train.loc[index,'model_txt2'] = row_txt2['model'].values[0]
    
    pan24_generative_authorship_smoke_train[['label_txt1', 'label_txt2']] = pan24_generative_authorship_smoke_train[['label_txt1', 'label_txt2']].astype(int)
    pan24_generative_authorship_smoke_train = pan24_generative_authorship_smoke_train[['id', 'label_txt1', 'label_txt2', 'model_txt1', 'model_txt2', 'text1', 'text2']]
    print(pan24_generative_authorship_smoke_train.info())
    pan24_generative_authorship_smoke_train_dict = pan24_generative_authorship_smoke_train.to_dict('records')
    utils.save_json(data=pan24_generative_authorship_smoke_train_dict, file_path=utils.INPUTS_DIR_PATH + "/pan24_generative_authorship_smoke_train.jsonl")

    #***** build pan24_generative_authorship_smoke_20240411_0_training truth file
    test_set_truth_lst = []
    for row in pan24_generative_authorship_smoke_train_dict:
        is_human_txt = 0
        if row['label_txt2'] == 1:
            is_human_txt = 1
        d = {'id': row['id'], 'is_human': is_human_txt}
        test_set_truth_lst.append(d)
    utils.save_json(data=test_set_truth_lst, file_path=utils.INPUTS_DIR_PATH + "/pan24_generative_authorship_smoke_train_truth.jsonl")


def main():
    #test_save_jsonl_data()
    #test_topics_distro()
    test_pan24_generative_authorship_smoke_20240411_0_training()

if __name__ == '__main__':
    main()
