"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""
import argparse
import datetime
import logging
import os
import sys
from typing import Callable, List

import pytz
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from mrc import MRC
from omegaconf import OmegaConf, dictconfig
from retrieval import DenseRetrieval, HybridRetrieval, SparseRetrieval
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, set_seed

logger = logging.getLogger(__name__)


def main(args):

    config = OmegaConf.load(f"./config/{args.config}.yaml")
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    if config.train.output_dir is None:
        trained_model = config.model.name_or_path
        if trained_model.startswith("./saved_models"):
            trained_model = trained_model.replace("./saved_models/", "")  # dropping "saved_models/" for sake of saving
        elif trained_model.startswith("saved_models"):
            trained_model = trained_model.replace("saved_models/", "")
        config.train.output_dir = os.path.join("predictions", trained_model, now_time)
        print(f"You can find the outputs in {config.train.output_dir}")
    training_args = TrainingArguments(**config.train)

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    # logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(config.utils.seed)

    datasets = load_from_disk(config.path.predict)
    print(datasets)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        from_tf=bool(".ckpt" in config.model.name_or_path),
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        config.model.name_or_path,
        from_tf=bool(".ckpt" in config.model.name_or_path),
    )
    print(f"Get the pretrained model {config.model.name_or_path}")

    reader = MRC(
        config,
        training_args,
        tokenizer,
        model,
    )

    # True일 경우 : run passage retrieval
    if config.retriever.type == "sparse":
        datasets = run_sparse_retrieval(
            tokenize_fn=tokenizer.tokenize,
            datasets=datasets,
            config=config,
        )
    elif config.retriever.type == "dense":
        datasets = run_dense_retrieval(datasets, config)
    elif config.retriever.type == "hybrid":
        datasets = run_hybrid_retrieval(tokenize_fn=tokenizer.tokenize, datasets=datasets, config=config)

    #### eval dataset & eval example - predictions.json 생성됨
    reader.predict(predict_dataset=datasets["validation"])


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    config: dictconfig.DictConfig,
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn,
        config=config,
    )
    if config.retriever.sparse.embedding_type == "tfidf":
        retriever.get_sparse_embedding()
        retriever.build_faiss()
    if config.retriever.faiss.use_faiss:
        df = retriever.retrieve_faiss(datasets["validation"])
    else:
        df = retriever.retrieve(datasets["validation"])

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_dense_retrieval(datasets, config):
    retriever = DenseRetrieval(config)
    retriever.get_dense_passage_embedding()
    df = retriever.retrieve(datasets["validation"])

    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_hybrid_retrieval(tokenize_fn, datasets, config):
    retriever = HybridRetrieval(tokenize_fn, config)
    df = retriever.retrieve(datasets["validation"])

    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="custom_config")
    args, _ = parser.parse_known_args()
    main(args)
