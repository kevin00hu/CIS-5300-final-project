#encoding=utf-8
from collections import Counter
from nltk.tokenize import word_tokenize
from typing import List
import gensim.downloader as api
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class dataset:
    def __init__(self, FNC_PATH = "./dataset/FNC-1", LIAR_PATH = "./dataset/LIAR", word2vec = True) -> None:
        self.FNC_PATH  = FNC_PATH
        self.LIAR_PATH = LIAR_PATH

        FNC_train, FNC_val, FNC_test = self.load_dataset(self.FNC_PATH)
        LIAR_train, LIAR_val, LIAR_test = self.load_dataset(self.LIAR_PATH, header = None)
        
        self.show_dataset_info(FNC_train, FNC_val, FNC_test, "FNC")
        self.show_dataset_info(LIAR_train, LIAR_val, LIAR_test, "LIAR")

        self.FNC_data = [FNC_train, FNC_val, FNC_test]
        self.LIAR_data = [LIAR_train, LIAR_val, LIAR_test]

        self.word2vec = word2vec
        if self.word2vec:
            global word2vec_model
            # statement (language to vector)
            word2vec_model = api.load("word2vec-google-news-300")
        self.preprocess()
#+------------------------------------------------------------------------------------------------------+
    # utils
    @staticmethod
    def load_dataset(path:str, header=0)->tuple:
        if os.path.exists(path):
            suffix = ".csv" if "LIAR" not in path else ".tsv"
            read_function = pd.read_csv if suffix != ".tsv" else pd.read_table

            train_path = os.path.join( path, f"train{suffix}" )
            val_path   = os.path.join( path, f"valid{suffix}" )
            test_path  = os.path.join( path, f"test{suffix}" )

            train_df = read_function(train_path, header = header)
            val_df   = read_function(val_path, header = header)
            test_df  = read_function(test_path, header = header)

            return (train_df, val_df, test_df)
        else:
            print(f"Target directory {path} not found.")

    @staticmethod
    def show_dataset_info(train_df, val_df, test_df, dataset_name:str):
        print(f"{dataset_name}")
        dataset_name = ["train", "validation", "test"]
        for dataset, name in zip([train_df, val_df, test_df], dataset_name):
            print( f"\t{name:10s} has {dataset.shape[0]:5d} rows and shapes {dataset.shape}." )

    # merged the dataset for easier process
    @staticmethod
    def merge_dataset(dataset_list:List[pd.DataFrame]):
        merged_dataset = pd.concat(dataset_list)
        dataset_length = [len(subdataset) for subdataset in dataset_list]
        return merged_dataset, dataset_length
    
    # split the merged df 
    @staticmethod
    def split_merged_dataset(merged_dataset:pd.DataFrame, length_list:List[int])->List[pd.DataFrame]:
        assert sum(length_list) == merged_dataset.shape[0], "incorrect length_list"
        train_length, val_length, test_length = length_list
        train_df = merged_dataset.iloc[:train_length]
        val_df   = merged_dataset.iloc[train_length: (train_length + val_length)]
        test_df  = merged_dataset.iloc[(train_length + val_length):]
        return train_df, val_df, test_df
    
    @staticmethod
    def get_word2vec(sentence):
        tokenized_sentence = word_tokenize(sentence.lower())
        word_vectors = [word2vec_model[word] for word in tokenized_sentence if word in word2vec_model]
        return np.array(word_vectors).mean(axis = 0)

#+------------------------------------------------------------------------------------------------------+
    # preprocess
    def preprocess(self):
        self.preprocess_FNC()
        self.preprocess_LIAR()

    #+-----------------------------------------------------------+
    # FNC part 
    def preprocess_FNC(self)->None:
        merged_FNC_df, length_list = self.merge_dataset(self.FNC_data)

        # drop unnamed column `Unnamed: 0`
        merged_FNC_df.drop(columns=["Unnamed: 0"])

        # change label to binary label
        label_binary_fnc = {
            "agree": 1,
            "disagree": 1,
            "discuss": 1,
            "unrelated": 0
        }

        merged_FNC_df['label'] = merged_FNC_df['Stance'].map(label_binary_fnc)

        merged_FNC_df['X'] = merged_FNC_df["Headline"] + merged_FNC_df["articleBody"]


        if self.word2vec:
            merged_FNC_df = merged_FNC_df.reset_index(drop=True)
            merged_FNC_df["statement_wv"] = merged_FNC_df["X"].apply(self.get_word2vec)

                # split a column of list to 300 columns 
            temp_df = pd.DataFrame(merged_FNC_df["statement_wv"].to_list(), columns=[f"vector_{i+1}" for i in range(300)])
            merged_FNC_df = pd.concat([merged_FNC_df, temp_df], axis = 1)

            self.FNC_data = self.split_merged_dataset(merged_FNC_df, length_list)

    # LIAR part
    def preprocess_LIAR(self)->None:
        merged_LIAR_df, length_list = self.merge_dataset(self.LIAR_data)

        # assign column names to the DF
        column_names = ["statement ID", "label", "statement", "subject(s)",
                    "speaker", "speaker's job title", 
                    "state", "party affiliation", 
                    "barely true", "false", "half true", "mostly true", "pants on fire", 
                    "context(location)"]
            # pants on fire: heavily false
        merged_LIAR_df.columns = column_names

        # convert 6 classes label to binary label
        label_binary_liar = {
            "false": 0,
            "half-true": 0,
            "pants-fire": 0,
            "barely-true": 0,
            "mostly-true": 1,
            "true": 1
        }

        merged_LIAR_df['label'] = merged_LIAR_df['label'].map(label_binary_liar)

        # subject column -> one-hot encoding
        merged_LIAR_df["subject(s)"] = merged_LIAR_df["subject(s)"].replace(np.nan, '')

        subject = merged_LIAR_df["subject(s)"].replace(np.nan, '') # There is nan in the column
        unique_subject = Counter([j for i in subject for j in i.split(",") if j])
        
            # subject thredhold 
        subject_thredhold = 20 

            # delete column count less than subject_thredhold
        for k in [i for i in unique_subject.keys()]:
            if unique_subject[k] < subject_thredhold:
                del unique_subject[k]

            # one-hot encoding
        for subject in unique_subject.keys():
            merged_LIAR_df[f"subject_{subject}"] = merged_LIAR_df["subject(s)"].apply(lambda x: subject in x)

            # assign subject_other to True for empty subject or subject count less than `subject_thredhold`
        merged_LIAR_df['subject_other'] = merged_LIAR_df[[i for i in merged_LIAR_df.columns if "subject_" in i]].sum(axis = 1) <= 0

        # state column
        state_thredhold = 20
    
            # state <= state_thredhold will be assigned "other" instead of original value
        label_count    = merged_LIAR_df[["label","state"]].groupby("state").count().sort_values(by="label")
        filtered_state = label_count[label_count["label"]>=state_thredhold].index
        merged_LIAR_df["raw_state"] = merged_LIAR_df["state"]
        merged_LIAR_df["state"]     = np.where(merged_LIAR_df["state"].isin(filtered_state), merged_LIAR_df["state"], "other")

        merged_LIAR_df = pd.concat([merged_LIAR_df, pd.get_dummies(merged_LIAR_df["state"])], axis = 1)

        merged_LIAR_df['party_other'] = ~merged_LIAR_df["party affiliation"].isin(["democrat", "republican"])
        merged_LIAR_df['party_democrat'] = merged_LIAR_df["party affiliation"] == "democrat"
        merged_LIAR_df['party_republican'] = merged_LIAR_df["party affiliation"] == "republican"

        drop_columns = ["statement ID", "subject(s)", "speaker", "speaker's job title", "state", "raw_state",
                                                    "barely true", "false", "half true", "mostly true", "pants on fire", 
                                                    "context(location)", "party affiliation"]

        if self.word2vec:
            merged_LIAR_df = merged_LIAR_df.reset_index(drop=True)

            # average word2vec
                # applying word2vec on word level and average them.
            merged_LIAR_df["statement_wv"] = merged_LIAR_df["statement"].apply(self.get_word2vec)

                # split a column of list to 300 columns 
            temp_df = pd.DataFrame(merged_LIAR_df["statement_wv"].to_list(), columns=[f"vector_{i+1}" for i in range(300)])
            merged_LIAR_df = pd.concat([merged_LIAR_df, temp_df], axis = 1)

            drop_columns.append("statement")
            drop_columns.append("statement_wv")

            # drop columns
        merged_LIAR_df = merged_LIAR_df.drop(columns = drop_columns)
        merged_LIAR_df = merged_LIAR_df *1 
        self.LIAR_data = self.split_merged_dataset(merged_LIAR_df, length_list)
#+------------------------------------------------------------------------------------------------------+
    # get data from the class
    def __call__(self, dataset:str="LIAR", type:str="train", all:bool=False):
        type_mapping = {
            "train": 0,
            "val": 1,
            "test": 2
        }

        return_dataset = self.LIAR_data if dataset=="LIAR" else self.FNC_data

        if all:
            return return_dataset
        return return_dataset[type_mapping[type]]
        
    
