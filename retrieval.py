import json
import logging
import os
import pickle
import sys
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from omegaconf import OmegaConf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from dataset.DPR_Dataset import DenseRetrievalDataset, DenseRetrievalValidDataset
from model.Retrieval.BertEncoder import BertEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetrieval:
    def __init__(
        self,
        config,
        data_path: Optional[str] = "./data/wikipedia_documents.json",
    ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.DPR.model.name)
        self.p_encoder = BertEncoder.from_pretrained(self.config.DPR.model.best_p_encoder_path)
        self.q_encoder = BertEncoder.from_pretrained(self.config.DPR.model.best_q_encoder_path)
        self.passage_embedding_vectors = None  # get_dense_passage_embedding()에서 생성

        self.data_path = data_path
        with open(data_path, "r") as f:
            wiki = json.load(f)
        self.contexts = list(dict.fromkeys(w["text"] for w in wiki.values()))
        self.ids = list(range(len(self.contexts)))

    def get_dense_passage_embedding(self):
        logger.info("Get dense passage embedding")
        if os.path.isfile(f"./saved_models/DPR/passage_embedding_vectors/p_embs_epoch_{self.config.DPR.model.saved_p_embs_epoch}.bin"):
            with open(f"./saved_models/DPR/passage_embedding_vectors/p_embs_epoch_{self.config.DPR.model.saved_p_embs_epoch}.bin", "rb") as f:
                self.passage_embedding_vectors = pickle.load(f)
            logger.info("Embedding pickle loaded")

        else:
            print("no saved pickle file")
            self.passage_embedding_vectors = []
            p_seqs = self.tokenizer(
                self.contexts,
                padding="max_length",
                max_length=self.config.DPR.tokenizer.max_context_length,
                truncation=True,
                return_tensors="pt",
            )
            # Dataset
            p_dataset = DenseRetrievalValidDataset(self.data_path, self.config.DPR.tokenizer.max_context_length, self.tokenizer)
            # DataLoader
            p_dataloader = torch.utils.data.DataLoader(p_dataset, batch_size=self.config.DPR.train.batch_size, shuffle=False, num_workers=4)
            # Make passage embedding
            for item in tqdm(p_dataloader):
                self.passage_embedding_vectors.extend(
                    self.p_encoder(
                        input_ids=item[0],
                        attention_mask=item[1],
                        token_type_ids=item[2],
                    )
                    .detach()
                    .numpy()
                )
                torch.cuda.empty_cache()
                del item
            self.passage_embedding_vectors = torch.Tensor(self.passage_embedding_vectors).squeeze()  # (56737, 768)

            with open("./saved_models/DPR/passage_embedding_vectors/p_embs.bin", "wb") as f:
                pickle.dump(self.passage_embedding_vectors, f)
            logger.info("Embedding pickle saved")

    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 10):
        """
        inference 할 때 사용
        """
        assert self.passage_embedding_vectors is not None, "Passage embedding vectors is None. Please run get_dense_passage_embedding() first."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_ids = self.get_relevant_doc(query_or_dataset, topk)

            for i in range(len(doc_scores)):
                print(f"top {i + 1} : {doc_scores[i]:.4f}")
                print(self.contexts[int(doc_ids[i])])

            return (doc_scores, [self.contexts[int(doc_ids[i])] for i in range(len(doc_ids))])

        elif isinstance(query_or_dataset, Dataset):
            correct_cnt = 0
            is_val = False
            total = []

            with timer("query exhaustive search"):
                doc_scores, doc_ids = self.get_relevant_doc_bulk(query_or_dataset["question"], topk)

            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.contexts[pid] for pid in doc_ids[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    is_val = True
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                    if tmp["original_context"] in tmp["context"]:
                        correct_cnt += 1
                total.append(tmp)

            cqas = pd.DataFrame(total)
            if is_val == True:
                print(f"Validation Accuracy: {correct_cnt / len(query_or_dataset) * 100:.2f}%")
                logger.info(f"Validation Accuracy: {correct_cnt / len(query_or_dataset) * 100:.2f}%")
            return cqas

    def get_relevant_doc(self, query, topk):
        q_seqs = self.tokenizer(
            [query], max_length=self.config.DPR.tokenizer.max_question_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        q_emb = self.q_encoder(**q_seqs)  # (1, 768)

        sim_scores = torch.matmul(q_emb, torch.transpose(self.passage_embedding_vectors, 0, 1))  # (1, 56737)

        rank = torch.argsort(sim_scores, dim=1, descending=True).squeeze()  # (56737)

        doc_ids = list(map(int, rank[:topk].tolist()))
        doc_scores = sim_scores.squeeze()[doc_ids].tolist()

        return doc_ids, doc_scores

    def get_relevant_doc_bulk(self, queries, topk):
        q_seqs = self.tokenizer(
            queries, max_length=self.config.DPR.tokenizer.max_question_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        q_dataset = DenseRetrievalDataset(
            input_ids=q_seqs["input_ids"], attention_mask=q_seqs["attention_mask"], token_type_ids=q_seqs["token_type_ids"]
        )
        q_dataloader = torch.utils.data.DataLoader(q_dataset, batch_size=1)

        doc_ids = []
        doc_scores = []
        for item in tqdm(q_dataloader):
            q_embs = self.q_encoder(input_ids=item[0], attention_mask=item[1], token_type_ids=item[2])
            for q_emb in q_embs:
                sim_scores = torch.matmul(q_emb, torch.transpose(self.passage_embedding_vectors, 0, 1))
                rank = torch.argsort(sim_scores, dim=0, descending=True)
                doc_ids.append(rank[:topk].tolist())
                doc_scores.append(sim_scores[rank[:topk]].tolist())
        return doc_scores, doc_ids


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        config,
    ) -> None:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab
            config:
                주요 args
                tfidf_num_features:
                    tfidf 벡터 차원 크기
                apply_lsa:
                    truncatedSVD를 이용해 추가로 tf-idf vectors에 LSA(latent semnatic analysis)를 적용할지 선택할 수 있습니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = config.path.data
        context_path = config.path.context
        args = config.retriever.sparse
        with open(context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidf_num_features = args.tfidf_num_features
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=self.tfidf_num_features,
        )
        self.apply_lsa = args.lsa
        self.lsa_vectorizer = None
        self.n_lsa_features = args.lsa_num_features
        if self.apply_lsa is True:
            self.lsa_vectorizer = TruncatedSVD(
                n_components=args.lsa_num_features,
                algorithm="arpack",
            )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.
        self.num_clusters = config.retriever.faiss.num_clusters
        self.topk = config.retriever.topk

    def get_sparse_embedding(self) -> None:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embeddings_{self.tfidf_num_features}.bin"
        tfidf_vectorizer_name = f"tfidf_vectorizer_{self.tfidf_num_features}.bin"
        lsa_vectorizer_name = f"lsa_vectorizer_{self.n_lsa_features}.bin"
        context_lsa_name = f"sparse_lsa_embeddings_{self.n_lsa_features}.bin"

        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidf_vectorizer_name)
        lsav_path = os.path.join(self.data_path, lsa_vectorizer_name)
        lsa_emd_path = os.path.join(self.data_path, context_lsa_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidf_vectorizer = pickle.load(file)
            print("Embedding pickle load.")

        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidf_vectorizer.fit_transform(self.contexts)
            print(f"tf-idf context embedding vector shape: {self.p_embedding.shape}")  # (56737, 50000)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidf_vectorizer, file)
            print("Embedding pickle saved.")

        if self.apply_lsa is True:
            if os.path.isfile(lsav_path):
                with open(lsav_path, "rb") as file:
                    self.lsa_vectorizer = pickle.load(file)
                with open(lsa_emd_path, "rb") as file:
                    self.lsa_embedding = pickle.load(file)
            else:
                self.lsa_embedding = self.lsa_vectorizer.fit_transform(self.p_embedding)
                with open(lsav_path, "wb") as file:
                    pickle.dump(self.lsa_vectorizer, file)
                with open(lsa_emd_path, "wb") as file:
                    pickle.dump(self.lsa_embedding, file)

    def build_faiss(self) -> None:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{self.num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            if self.apply_lsa is True:
                p_emb = self.lsa_embedding.astype(np.float32).toarray()
            else:
                p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            quantizer = faiss.IndexFlatL2(emb_dim)
            self.indexer = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, self.num_clusters, faiss.METRIC_L2)
            self.indexer.train(p_emb)  # 원래는 벡터 분포를 알기 위해. IndexFlatL2는 train이 필요 없음
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(self, query_or_dataset: Union[str, Dataset]) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(self.topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(self.topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"])
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidf_vectorizer.transform([query])
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[: self.topk]
        doc_indices = sorted_result.tolist()[: self.topk]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(self, queries: List) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.

        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        if self.apply_lsa:
            query_vec = self.lsa_vectorizer.transform(queries)
        else:
            query_vec = self.tfidf_vectorizer.transform(queries)
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[: self.topk])
            doc_indices.append(sorted_result.tolist()[: self.topk])
        return doc_scores, doc_indices

    def retrieve_faiss(self, query_or_dataset: Union[str, Dataset]) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(query_or_dataset)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(self.topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(self.topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(self, query: str) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidf_vectorizer.transform([query])
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, self.topk)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(self, queries: List) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        if self.apply_lsa is True:
            query_vecs = self.lsa_vectorizer.transform(queries)
        else:
            query_vecs = self.tfidf_vectorizer.transform(queries)
        assert np.sum(query_vecs) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, self.topk)

        return D.tolist(), I.tolist()


class HybridRetrieval:
    """
    Sparse Retrieval Score에 Dense Retrieval Score를 더해주어 Reranking 수행
    """

    def __init__(self, tokenize_fn, config):
        self.dense_retriever = DenseRetrieval(config)
        self.sparse_retriever = SparseRetrieval(tokenize_fn=tokenize_fn, config=config)

        self.queries = None

    def get_sparse_score(self):
        self.sparse_retriever.get_sparse_embedding()
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc_bulk(queries=self.queries)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", metavar="./data/train_dataset", type=str, help="")
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument("--context_path", metavar="wikipedia_documents", type=str, help="")
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")

    parser.add_argument("--config", "-c", type=str, default="base_config")
    args = parser.parse_args()

    config = OmegaConf.load(f"./config/{args.config}.yaml")

    # logging 설정
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
        with open("./logs/DPR_logs.log", "w+") as f:
            f.write("***** Log file Start *****\n")
    logging.basicConfig(
        filename="./logs/DPR_logs.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.info("***** retrieval.py *****")
    # Test sparse
    org_dataset = load_from_disk("./data/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    if config.retrieval.type == "sparse":
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False,
        )

        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
    elif config.retrieval.type == "dense":
        retriever = DenseRetrieval(config)
        retriever.get_dense_passage_embedding()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, topk=config.retrieval.topk)
            df["correct"] = df["original_context"] == df["context"]
            df.to_csv(".results.csv", index=False, encoding="utf-8")  # 💥
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
