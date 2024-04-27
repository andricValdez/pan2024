import json
import random
import glob
import pprint
import os
import joblib
import argparse
import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
OUTPUT_DIR_PATH = ROOT_DIR + '/outputs/'
INPUTS_DIR_PATH = ROOT_DIR + '/inputs/'
ROOT_PAN24_PATH = ROOT_DIR + '/datasets/pan24/'
ROOT_SEMEVAL24_PATH = ROOT_DIR + '/datasets/semeval24/'


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