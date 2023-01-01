import argparse
import datetime
import logging
import os
import sys

import pytz
import torch
import wandb
from dataset.DPR_Dataset import DenseRetrievalTrainDataset, DenseRetrievalValidDataset
from model.Retrieval.BertEncoder import BertEncoder
from omegaconf import OmegaConf
from trainer.DenseRetrievalTrainer import DenseRetrievalTrainer
from transformers import AutoTokenizer, TrainingArguments, set_seed

logger = logging.getLogger(__name__)


def main(config):
    config = OmegaConf.load(f"./config/{args.config}.yaml")
    # wandb 설정
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    run_id = f"{config.wandb.name}_{now_time}"
    wandb.init(
        entity=config.wandb.team,
        project=config.wandb.project,
        group=config.model.name,
        id=run_id,
        tags=config.wandb.tags,
    )

    config.DPR.train.update(config.DPR.optimizer)
    if config.DPR.train.output_dir is None:
        config.DPR.train.output_dir = os.path.join("saved_models/DPR", config.DPR.model.name, run_id)

    # logging 설정
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
        with open("./logs/DPR_logs.log", "w+") as f:
            f.write("***** Log file Start *****\n")
    LOG_FORMAT = "%(asctime)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    file_handler = logging.FileHandler("./logs/DPR_logs.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("config", config)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(config.utils.seed)

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(config.DPR.model.name)

    # 데이터셋
    train_dataset = DenseRetrievalTrainDataset(
        data_path=config.DPR.path.train,
        max_context_length=config.DPR.tokenizer.max_context_length,
        max_question_length=config.DPR.tokenizer.max_question_length,
        tokenizer=tokenizer,
        hard_negative=config.DPR.train.hard_negative,
    )
    valid_dataset = DenseRetrievalValidDataset(
        data_path=config.DPR.path.valid,
        max_context_length=config.DPR.tokenizer.max_context_length,
        tokenizer=tokenizer,
    )
    logger.info(f"  train_dataset: {len(train_dataset)} | valid_dataset: {len(valid_dataset)}")

    # 모델
    logger.info(f"  Encoder model: {config.DPR.model.name}")
    p_encoder = BertEncoder.from_pretrained(config.DPR.model.name)
    q_encoder = BertEncoder.from_pretrained(config.DPR.model.name)
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    # 학습
    training_args = TrainingArguments(
        output_dir=config.DPR.train.output_dir,
        evaluation_strategy="epoch",
        learning_rate=config.DPR.optimizer.learning_rate,
        per_device_train_batch_size=config.DPR.train.batch_size,
        per_device_eval_batch_size=config.DPR.train.batch_size,
        num_train_epochs=config.DPR.train.num_train_epochs,
        weight_decay=config.DPR.optimizer.weight_decay,
        gradient_accumulation_steps=config.DPR.optimizer.gradient_accumulation_steps,
    )
    training_args.report_to = ["wandb"]

    trainer = DenseRetrievalTrainer(training_args, config, tokenizer, p_encoder, q_encoder, train_dataset, valid_dataset)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="custom_config")
    args, _ = parser.parse_known_args()
    main(args)
