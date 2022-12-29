import json
import os
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

        df_wiki = pd.read_json("./data/wikipedia_documents.json").T
        df_wiki["text"] = df_wiki["text"].str.replace("[ \\\\n+|\\n+]+", " ", regex=True)
        """corpus_source, url, domain, title, author, html, document_id 컬럼들은 사용하지 않는다고 가정하고 text 컬럼의 중복만 제거"""
        df_wiki = df_wiki.drop_duplicates(subset=["text"])["text"]
        self.corpus = pd.DataFrame({"context": [example for example in df_wiki]})
        print("corpus len : ", len(self.corpus))
        self.tokenized_corpus = [doc.split(" ") for doc in self.corpus["context"]]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.preprocessed_data = self.preprocess(data_path)

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
            self.preprocessed_data[2]["input_ids"][idx],  # hard_negative
            self.preprocessed_data[2]["attention_mask"][idx],  # hard_negative
            self.preprocessed_data[2]["token_type_ids"][idx],  # hard_negative
        )

    def preprocess(self, data_path):
        data = load_from_disk(data_path)

        data_question = data["question"]
        data_context = data["context"]

        if True:  # hard negative 설정
            print("hard negative 추출 시작")
            if os.path.exists("hard_negative.csv"):
                print("hard negative 파일이 존재하여 불러옵니다.")
                df_hard_negative = pd.read_csv("hard_negative.csv")
            else:
                hard_negative = [[], [], []]
                print("hard negative 추출 중...")
                for context in tqdm(data_context, total=len(data_context)):
                    passages = self.hard_negative(context, self.corpus["context"], k=4)
                    for i in range(3):
                        try:
                            hard_negative[i].append(passages.iloc[i])
                        except:
                            hard_negative[i].append("None")

                print("hard negative passages 토큰화 중...")
                try:
                    df_hard_negative = pd.DataFrame(
                        {
                            "passage_top1": hard_negative[0],
                            "enc_passage_top1": self.tokenizer(
                                hard_negative[0], padding="max_length", max_length=self.max_context_length, truncation=True, return_tensors="pt"
                            )["input_ids"],
                            "passage_top2": hard_negative[1],
                            "enc_passage_top2": self.tokenizer(
                                hard_negative[1], padding="max_length", max_length=self.max_context_length, truncation=True, return_tensors="pt"
                            )["input_ids"],
                            "passage_top3": hard_negative[2],
                            "enc_passage_top3": self.tokenizer(
                                hard_negative[2], padding="max_length", max_length=self.max_context_length, truncation=True, return_tensors="pt"
                            )["input_ids"],
                        }
                    )
                except:
                    df_hard_negative = pd.DataFrame(
                        {
                            "passage_top1": hard_negative[0],
                            "passage_top2": hard_negative[1],
                            "passage_top3": hard_negative[2],
                        }
                    )
                print("hard negative passages 토큰화 완료!")
                df_hard_negative.to_csv("hard_negative.csv", index=False)
                print("hard_negative.csv 저장 완료!")
            hn_seqs = self.tokenizer(
                list(df_hard_negative["passage_top1"]), padding="max_length", max_length=self.max_context_length, truncation=True, return_tensors="pt"
            )

        p_seqs = self.tokenizer(data_context, padding="max_length", max_length=self.max_context_length, truncation=True, return_tensors="pt")
        q_seqs = self.tokenizer(data_question, padding="max_length", max_length=self.max_question_length, truncation=True, return_tensors="pt")

        if True:  # hard negative 설정
            return p_seqs, q_seqs, hn_seqs
        else:
            return p_seqs, q_seqs

    def hard_negative(self, query, corpus, k=3):
        tokenized_query = query.split(" ")

        passages = self.bm25.get_top_n(tokenized_query, corpus, n=k)
        # doc_scores = self.bm25.get_scores(tokenized_query)
        # indices = np.argsort(-doc_scores)[:k]
        # scores = doc_scores[np.argsort(-doc_scores)[:k]]

        # df = pd.DataFrame({"index": indices, "passage": passages, "score": scores})
        df = pd.DataFrame({"passage": passages})
        try:
            """\n, \\n, 공백까지 모두 지우고 비교하도록 만듬"""
            temp = df[df["passage"].str.replace("[ \\\\n|\\n]+", "", regex=True) == re.sub("[ \\\\n|\\n]+", "", query)]["passage"]
            df = df[df["passage"] != temp[0]]  # 정답 passage 제외
            return df["passage"].iloc[:]  # 정답 passage를 제외한 passage 중 가장 유사도가 높은 passage
        except:
            print("@ query", query[:100])
            # print("@@ df", df)
            return df["passage"].iloc[1:]  # 정답 passage를 제외한 passage 중 가장 유사도가 높은 passage


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
