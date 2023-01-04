import pandas as pd
import os
from collections import Counter
from natsort import natsorted
from tqdm import tqdm

class Ensemble:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filenames = os.listdir(filepath)
        self.filenames = natsorted(self.filenames)
        self.output_list = []

        output_path = [filepath + filename for filename in self.filenames]
        self.csv_nums = len(output_path)

        for path in output_path:
            self.output_list.append(pd.read_csv(path).groupby(["user"])['item'].apply(list).reset_index())
    
    def merge_item_hard(self, csv_list):
        first = csv_list[0]
        print("...calculating...")
        for i in tqdm(range(len(first))):
            for csv in csv_list[1:]:
                first['item'][i].extend(csv['item'][i])
        return first

    def merge_item_weighted1(self, csv_list, weight):
        first = csv_list[0]
        first['item'] = first['item'].apply(item2score, score = weight[0])

        print("...calculating...")
        for i in tqdm(range(len(first))):
            for csv, wei in zip(csv_list[1:], weight[1:]):
                for c in csv['item'][i]:
                    if first['item'][i].get(c):
                        first['item'][i][c] += wei
                    else:
                        first['item'][i][c] = wei
        return first

    def hard(self):
        merge_csv = self.merge_item_hard(self.output_list)
        merge_csv['item'] = merge_csv['item'].apply(topten)
        return merge_csv.explode(column=['item'])

    def weighted(self, weight: list):
        merge_csv = self.merge_item_weighted1(self.output_list, weight)
        merge_csv['item'] = merge_csv['item'].apply(topten_weighted)
        merge_csv = merge_csv.explode(column=['item'])
        return merge_csv

def item2score(item, score):
    return {i:score for i in item}

def topten(i):
    return [ key for key, _ in Counter(i).most_common(10) ]

def topten_weighted(i):
    temp = sorted(i.items(), key=lambda x: x[1], reverse=True)[:10]
    temp = [t[0] for t in temp]
    return temp