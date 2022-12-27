import json
import re

import numpy as np
import pandas as pd
from datasets import load_from_disk
from rank_bm25 import BM25Okapi
from torch.utils.data import Dataset
from tqdm import tqdm


class DenseRetrievalTrainDataset(Dataset):
    """
    Dense Passage Retrieval Train을 위한 Dataset
    """

    def __init__(self, data_path, max_context_length, max_question_length, tokenizer):
        self.max_context_length = max_context_length
        self.max_question_length = max_question_length
        self.tokenizer = tokenizer
        self.preprocessed_data = self.preprocess(data_path)

        df_wiki = pd.read_json("./data/wikipedia_documents.json").T
        self.corpus = pd.DataFrame({"context": list(set([re.sub("[\\\\n|\\n]+", " ", example) for example in df_wiki["text"]]))})
        self.bm25 = BM25Okapi(self.corpus)

    def __len__(self):
        return self.preprocessed_data[0]["input_ids"].size(0)

    def __getitem__(self, idx):
        return (
            self.preprocessed_data[0]["input_ids"][idx],  # context
            self.preprocessed_data[0]["attention_mask"][idx],  # context
            self.preprocessed_data[0]["token_type_ids"][idx],  # context
            self.preprocessed_data[1]["input_ids"][idx],  # question
            self.preprocessed_data[1]["attention_mask"][idx],  # question
            self.preprocessed_data[1]["token_type_ids"][idx],  # question
            self.preprocessed_data[2][idx],  # hard_negative
        )

    def preprocess(self, data_path):
        data = load_from_disk(data_path)

        data_question = data["question"]
        data_context = data["context"]

        p_seqs = self.tokenizer(data_context, padding="max_length", max_length=self.max_context_length, truncation=True, return_tensors="pt")
        q_seqs = self.tokenizer(data_question, padding="max_length", max_length=self.max_question_length, truncation=True, return_tensors="pt")

        print("######## best sim passage")
        hard_negative = []
        for context_idx, context in enumerate(tqdm(data_context, total=len(data_context))):
            hard_negative.append(self.hard_negative(context, self.tokenizer, self.corpus))

        return p_seqs, q_seqs, hard_negative

    def hard_negative(self, query, tokenizer, corpus, k=3):
        tokenized_query = tokenizer(query)

        passages = self.bm25.get_top_n(tokenized_query, corpus, n=k)
        # doc_scores = self.bm25.get_scores(tokenized_query)
        # indices = np.argsort(-doc_scores)[:k]
        # scores = doc_scores[np.argsort(-doc_scores)[:k]]

        # df = pd.DataFrame({"index": indices, "passage": passages, "score": scores})
        df = pd.DataFrame({"passage": passages})
        temp = df[df["passage"] == query]["passage"]
        df = df[df["passage"] != temp[0]]  # 정답 passage 제외

        return df["passage"].iloc[0]  # 정답 passage를 제외한 passage 중 가장 유사도가 높은 passage


class DenseRetrievalValidDataset(Dataset):
    """
    Dense Passage Retrieval Validation을 위한 Dataset
    """

    def __init__(self, data_path, max_context_length, tokenizer):
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.preprocessed_data = self.preprocess(data_path)

    def __len__(self):
        return self.preprocessed_data["input_ids"].size(0)

    def __getitem__(self, idx):
        return (
            self.preprocessed_data["input_ids"][idx],  # context
            self.preprocessed_data["attention_mask"][idx],  # context
            self.preprocessed_data["token_type_ids"][idx],  # context
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


class DenseRetrievalDataset(Dataset):
    """
    inference 시 받을 queries을 위한 데이터셋
    """

    def __init__(self, input_ids, attention_mask, token_type_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
        )
