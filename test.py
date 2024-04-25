
# import json module
import json
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
OUTPUT_DIR_PATH = ROOT_DIR + '/outputs'
INPUTS_DIR_PATH = ROOT_DIR + '/inputs'
ROOT_PAN24_PATH = ROOT_DIR + '/datasets/pan24'
ROOT_SEMEVAL24_PATH = ROOT_DIR + '/datasets/semeval24'

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