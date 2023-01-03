import pandas as pd
import os
from collections import Counter

class Ensemble:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filenames = os.listdir(filepath)
        self.output_list = []

        output_path = [filepath + filename for filename in self.filenames]
        self.csv_nums = len(output_path)

        for path in output_path:
            self.output_list.append(pd.read_csv(path).groupby(["user"])['item'].apply(list).reset_index())
        # for filename, output in zip(self.filenames, self.output_list):
        #     self.output_df[filename] = output

    def merge_item(self, csv_list):
        first = csv_list[0]
        for i in range(len(first)):
            for csv in csv_list[1:]:
                first['item'][i].extend(csv['item'][i])
        return first
    
    def topten(self, i):
        return [ key for key, _ in Counter(i).most_common(10) ]

    def hard(self):
        merge_csv = self.merge_item(self.output_list)
        merge_csv['item'] = merge_csv['item'].apply(self.topten)
        return merge_csv.explode(column=['item'])

            

    def weighted(self, weight: list):
        raise NotImplementedError
        
