import pandas as pd
from functools import reduce
import time
import re
import os
from tqdm import tqdm
import yaml
from yaml.loader import SafeLoader
from datasets import MRDataset


class Preprocess:
    def __init__(self, data_load_path):
        self.data_load_path = data_load_path

        self.org_director_data = pd.read_csv(
            os.path.join(self.data_load_path, "directors.tsv"), sep="\t"
        )
        self.org_year_data = pd.read_csv(
            os.path.join(self.data_load_path, "years.tsv"), sep="\t"
        )
        self.org_writer_data = pd.read_csv(
            os.path.join(self.data_load_path, "writers.tsv"), sep="\t"
        )
        self.org_title_data = pd.read_csv(
            os.path.join(self.data_load_path, "titles.tsv"), sep="\t"
        )
        self.org_genre_data = pd.read_csv(
            os.path.join(self.data_load_path, "genres.tsv"), sep="\t"
        )
        self.org_train_data = pd.read_csv(
            os.path.join(self.data_load_path, "train_ratings.csv")
        )

    def __grouping_data(self, data: pd.DataFrame, column_name: str) -> pd.DataFrame:
        return data.groupby("item")[column_name].apply("/".join).reset_index(drop=False)

    def __merge_data(self):
        self.director_data = self.__grouping_data(self.org_director_data, "director")
        self.writer_data = self.__grouping_data(self.org_writer_data, "writer")
        self.genre_data = self.__grouping_data(self.org_genre_data, "genre")
        datas = [
            self.org_title_data,
            self.org_year_data,
            self.director_data,
            self.writer_data,
            self.genre_data,
        ]
        self.data = reduce(
            lambda left, right: pd.merge(left, right, on="item", how="left"), datas
        )
        return self.data

    def __preprocessing_year_title(self):
        self.data["year"] = self.data["title"].apply(
            lambda x: re.search("\(\d{4}", x).group()[1:]
        )
        self.data["title"] = self.data["title"].apply(
            lambda x: re.sub("\(\d+-{0,1}\d*\)", "", x)
        )
        self.data["title"] = [
            row.split(',')[1] + ' ' + row.split(',')[0] if ', The' in row else row
            for row in self.data['title']
        ]
        self.data["title"] = self.data["title"].apply(
            lambda x: x.replace(',', '').strip()
        )

        self.data["title"] = self.data["title"].apply(
            lambda x: x.replace("\'09\"", '-09-')
        )
        self.data["title"] = self.data["title"].apply(lambda x: x.replace('\"', ''))
        self.data["title"] = self.data["title"].apply(lambda x: x.replace("'", "\'"))
        self.data["title"] = self.data["title"].apply(lambda x: re.sub(' +', ' ', x))
        return self.data

    def __preprocessing_NA(self):
        self.data["director"] = self.data["director"].fillna("others")
        self.data["writer"] = self.data["writer"].fillna("others")
        return self.data

    def preprocessing(self):
        data = self.__merge_data()
        data = self.__preprocessing_year_title()
        data = self.__preprocessing_NA()
        return data


if __name__ == "__main__":
    with open('./yaml/datasetting.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)
    data_load_path = config['data_load_path']
    data_save_path = config['data_save_path']
    atomic_label = config['atomic_label']
    atomic_cut = config['atomic_cut']

    preprocess = Preprocess(data_load_path)

    # generate inter_label data
    if not os.path.isfile(os.path.join(data_save_path, 'train_ratings.csv')):
        inter_label = pd.read_csv(os.path.join(data_load_path, 'train_ratings.csv'))
        inter_label['label'] = 1
        inter_label.to_csv(
            os.path.join(data_save_path, 'train_labels.csv'),
            sep='\t',
            index=False,
            encoding='utf-8-sig',
        )

    # generate item data
    if not os.path.isfile(os.path.join(data_save_path, 'item.csv')):
        item = preprocess.preprocessing()
        item.to_csv(
            os.path.join(data_save_path, 'item.csv'),
            sep='\t',
            index=False,
            encoding='utf-8-sig',
        )

    # generate atomic data
    if atomic_label:
        atomic_save_path = os.path.join(data_save_path, f'MR_label_cut_{atomic_cut}')
    else:
        atomic_save_path = os.path.join(data_save_path, f'MR_cut_{atomic_cut}')
    if not os.path.isdir(atomic_save_path):
        mrds = MRDataset(data_save_path, atomic_save_path, atomic_label, atomic_cut)
        mrds.convert_inter()
        mrds.convert_item()
