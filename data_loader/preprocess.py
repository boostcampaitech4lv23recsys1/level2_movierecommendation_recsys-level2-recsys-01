import pandas as pd
from functools import reduce
import time
import re
import os
from tqdm import tqdm


class Preprocess:
    def __init__(self, config):
        self.config = config
        self.data_path = config["preprocess"]["data_dir"]

        self.org_director_data = pd.read_csv(
            os.path.join(self.data_path, "directors.tsv"), sep="\t"
        )
        self.org_year_data = pd.read_csv(
            os.path.join(self.data_path, "years.tsv"), sep="\t"
        )
        self.org_writer_data = pd.read_csv(
            os.path.join(self.data_path, "writers.tsv"), sep="\t"
        )
        self.org_title_data = pd.read_csv(
            os.path.join(self.data_path, "titles.tsv"), sep="\t"
        )
        self.org_genre_data = pd.read_csv(
            os.path.join(self.data_path, "genres.tsv"), sep="\t"
        )
        self.org_train_data = pd.read_csv(
            os.path.join(self.data_path, "train_ratings.csv")
        )

    def __grouping_data(self, data: pd.DataFrame, column_name: str) -> pd.DataFrame:
        return data.groupby("item")[column_name].apply("/".join).reset_index(drop=False)

    def __merge_data(self):
        self.director_data = self.__grouping_data(self.org_director_data, "director")
        self.writer_data = self.__grouping_data(self.org_writer_data, "writer")
        self.genre_data = self.__grouping_data(self.org_genre_data, "genre")
        datas = [
            self.org_train_data,
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

    def __preprocessing_timestamp(self):
        self.data["watch_year"] = self.data["time"].apply(
            lambda x: time.strftime("%Y", time.localtime(x))
        )
        self.data["watch_month"] = self.data["time"].apply(
            lambda x: time.strftime("%m", time.localtime(x))
        )
        self.data["watch_day"] = self.data["time"].apply(
            lambda x: time.strftime("%d", time.localtime(x))
        )
        self.data["watch_hour"] = self.data["time"].apply(
            lambda x: time.strftime("%H", time.localtime(x))
        )
        return self.data

    def __preprocessing_NA(self):
        self.data["director"] = self.data["director"].fillna("others")
        self.data["writer"] = self.data["writer"].fillna("others")
        return self.data

    def preprocessing(self):
        data = self.__merge_data()
        data = self.__preprocessing_year_title()
        data = self.__preprocessing_timestamp()
        data = self.__preprocessing_NA()
        return data
