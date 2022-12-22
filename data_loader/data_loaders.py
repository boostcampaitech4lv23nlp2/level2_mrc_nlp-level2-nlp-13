import os
import pickle
import re
from abc import ABC, abstractmethod

import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import numpy as np
import json

from datasets import load_from_disk


class DenseRetrievalTrainDataset(Dataset):
    """
    Dense Passage Retrieval을 위한 Dataset
    """

    def __init__(self, data_path, max_context_length, max_question_length, tokenizer):
        self.max_context_length = max_context_length
        self.max_question_length = max_question_length
        self.tokenizer = tokenizer
        self.preprocessed_data = self.preprocess(data_path)

    def __len__(self):
        return self.preprocessed_data[0]["input_ids"].size(0)

    def __getitem__(self, idx):
        return (
            self.preprocessed_data[0]["input_ids"][idx],
            self.preprocessed_data[0]["attention_mask"][idx],
            self.preprocessed_data[0]["token_type_ids"][idx],
            self.preprocessed_data[1]["input_ids"][idx],
            self.preprocessed_data[1]["attention_mask"][idx],
            self.preprocessed_data[1]["token_type_ids"][idx],
        )

    def preprocess(self, data_path):
        data = load_from_disk(data_path)

        data_question = data["question"]
        data_context = data["context"]

        p_seqs = self.tokenizer(data_context, padding="max_length", max_length=self.max_context_length, truncation=True, return_tensors="pt")
        q_seqs = self.tokenizer(data_question, padding="max_length", max_length=self.max_question_length, truncation=True, return_tensors="pt")

        return p_seqs, q_seqs


class DenseRetrievalValidDataset(Dataset):
    def __init__(self, data_path, max_context_length, tokenizer):
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.preprocessed_data = self.preprocess(data_path)

    def __len__(self):
        return self.preprocessed_data["input_ids"].size(0)

    def __getitem__(self, idx):
        return (
            self.preprocessed_data["input_ids"][idx],
            self.preprocessed_data["attention_mask"][idx],
            self.preprocessed_data["token_type_ids"][idx],
        )

    def preprocess(self, data_path):
        if data_path in "./data/train_dataset/validation":
            data = load_from_disk(data_path)
            data_context = data["context"]
            p_seqs = self.tokenizer(data_context, padding="max_length", max_length=self.max_context_length, truncation=True, return_tensors="pt")
            return p_seqs

        elif data_path in "./data/wikipedia_documents.json":
            with open(data_path, "r") as f:
                wiki = json.load(f)
            data_context = list(dict.fromkeys(w["text"] for w in wiki.values()))
            p_seqs = self.tokenizer(data_context, padding="max_length", max_length=self.max_context_length, truncation=True, return_tensors="pt")
            return p_seqs
