import datetime
import logging

import pytz
import torch

import wandb

from model.Retrieval.BertEncoder import BertEncoder

from data_loader.data_loaders import DenseRetrievalTrainDataset, DenseRetrievalValidDataset
from transformers import AutoTokenizer, TrainingArguments
from trainer.retrieval_trainer import RetrievalTrainer


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥device : ", device)

    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H:%M:%S")
    if config.wandb.use == True:
        wandb.init(
            entity=config.wandb.team_account_name,
            project=config.wandb.project_repo,
            name=f"{config.wandb.name}_{config.wandb.info}_{now_time}",
        )

    print("🔥 get dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config.retrieval.model.name)

    train_dataset = DenseRetrievalTrainDataset(
        data_path=config.path.retrieval_train_path,
        max_context_length=config.tokenizer.max_context_length,
        max_question_length=config.tokenizer.max_question_length,
        tokenizer=tokenizer,
    )
    valid_dataset = DenseRetrievalValidDataset(
        data_path=config.path.retrieval_valid_path,
        max_context_length=config.tokenizer.max_context_length,
        tokenizer=tokenizer,
    )

    print("🔥 get model...")
    p_encoder = BertEncoder.from_pretrained(config.retrieval.model.name)
    q_encoder = BertEncoder.from_pretrained(config.retrieval.model.name)
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    print("🔥 start training...")

    training_args = TrainingArguments(
        output_dir=config.path.retrieval_save_path,
        evaluation_strategy="epoch",
        learning_rate=config.retrieval.train.learning_rate,
        per_device_train_batch_size=config.retrieval.train.batch_size,
        per_device_eval_batch_size=config.retrieval.train.batch_size,
        num_train_epochs=config.retrieval.train.max_epoch,
        weight_decay=config.retrieval.train.weight_decay,
        gradient_accumulation_steps=config.retrieval.train.gradient_accumulation_steps,
    )

    retrieval_trainer = RetrievalTrainer(training_args, config, tokenizer, p_encoder, q_encoder, train_dataset, valid_dataset)
    retrieval_trainer.train()
