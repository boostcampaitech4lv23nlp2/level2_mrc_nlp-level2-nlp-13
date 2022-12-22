import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange

import os
import numpy as np
from datasets import load_from_disk
from utils.logger import get_logger

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

logger = get_logger("./logs/dense_retrieval.log")


class RetrievalTrainer:
    def __init__(self, args, config, tokenizer, p_encoder, q_encoder, train_dataset, valid_dataset):
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def configure_optimizers(self, optimizer_grouped_parameters, config):
        if config.retrieval.train.optimizer == "adamw":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.retrieval.train.learning_rate)
        return optimizer

    def train_per_epoch(self, epoch_iterator: DataLoader, optimizer, scheduler):
        batch_loss = 0

        for _, batch in enumerate(epoch_iterator):
            self.p_encoder.train()
            self.q_encoder.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }

            p_outputs = self.p_encoder(**p_inputs)  # (batch_size, 768)
            q_outputs = self.q_encoder(**q_inputs)  # (batch_size, 768)

            # Calculate Similarity score & loss
            sim_score = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, batch_size)

            # target = position of positive sample = diagonal
            targets = torch.arange(0, self.args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(sim_score, dim=1)
            loss = F.nll_loss(sim_scores, targets)

            loss.backward()
            optimizer.step()
            # scheduler.step()
            self.q_encoder.zero_grad()
            self.p_encoder.zero_grad()

            batch_loss += loss.detach().cpu().numpy()
        torch.cuda.empty_cache()
        return batch_loss / len(epoch_iterator)

    def valid_per_epoch(self, valid_dataloader: DataLoader):
        logger.info("*** Validating ***")
        # passage embedding 생성
        p_embs = []
        with torch.no_grad():
            epoch_iterator = tqdm(valid_dataloader, desc="Iteration", position=0, leave=True)
            self.p_encoder.eval()

            for _, batch in enumerate(epoch_iterator):
                batch = tuple(t.cuda() for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                outputs = self.p_encoder(**p_inputs).to("cpu").numpy()
                p_embs.extend(outputs)

            p_embs = torch.Tensor(p_embs)  # (num_passage, 768)

        # Question Embeddig 생성
        top_1, top_5, top_10, top_25, top_50, top_100 = 0, 0, 0, 0, 0, 0

        valid_data = load_from_disk("./data/train_dataset/validation")
        valid_question = valid_data["question"]

        with torch.no_grad():
            self.q_encoder.eval()
            for idx in tqdm(range(len(valid_question))):
                query = valid_question[idx]

                q_seq = self.tokenizer(
                    [query],
                    max_length=self.config.retrieval.tokenizer.max_question_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to("cuda")
                q_emb = self.q_encoder(**q_seq).to("cpu")

                # Cosine Similarity 계산
                dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
                rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()  # (1,num_passage) -> (num_passage)

                # 성능 평가
                if idx == rank[0]:
                    top_1 += 1
                if idx in rank[:5]:
                    top_5 += 1
                if idx in rank[:10]:
                    top_10 += 1
                if idx in rank[:25]:
                    top_25 += 1
                if idx in rank[:50]:
                    top_50 += 1
                if idx in rank[:100]:
                    top_100 += 1
        return (
            top_1 / len(valid_question) * 100,
            top_5 / len(valid_question) * 100,
            top_10 / len(valid_question) * 100,
            top_25 / len(valid_question) * 100,
            top_50 / len(valid_question) * 100,
            top_100 / len(valid_question) * 100,
        )

    def train(self):
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  batch size  = %d", self.args.per_device_train_batch_size)

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.per_device_train_batch_size, drop_last=True)
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.args.per_device_eval_batch_size)

        best_top_1, best_top_5, best_top_10, best_top_25, best_top_50, best_top_100 = 0, 0, 0, 0, 0, 0

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            },
            {
                "params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = self.configure_optimizers(optimizer_grouped_parameters, self.config)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=t_total)

        # Train
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            # train per epoch
            train_loss = self.train_per_epoch(epoch_iterator, optimizer, scheduler)

            # valid per epoch
            top_1_acc, top_5_acc, top_10_acc, top_25_acc, top_50_acc, top_100_acc = self.valid_per_epoch(valid_dataloader)

            logger.info("***** Validation Result *****")
            logger.info(
                f"epoch: {epoch} | train loss: {train_loss:.4f} | top_1_acc: {top_1_acc:.2f} | top_5_acc: {top_5_acc:.2f} | top_10_acc: {top_10_acc:.2f} | top_25_acc: {top_25_acc:.2f} | top_50_acc: {top_50_acc:.2f} | top_100_acc: {top_100_acc:.2f}"
            )

            scheduler.step()

            # Save model 나중에 validation해서 최고 기록 낸 모델 저장으로 수정
            if top_1_acc > best_top_1:
                best_top_1 = top_1_acc
                self.q_encoder.save_pretrained(self.config.path.retrieval_save_path + "/q_encoder_best_top_1")
                self.p_encoder.save_pretrained(self.config.path.retrieval_save_path + "/p_encoder_best_top_1")
            if top_5_acc > best_top_5:
                best_top_5 = top_5_acc
                self.q_encoder.save_pretrained(self.config.path.retrieval_save_path + "/q_encoder_best_top_5")
                self.p_encoder.save_pretrained(self.config.path.retrieval_save_path + "/p_encoder_best_top_5")
            if top_10_acc > best_top_10:
                best_top_10 = top_10_acc
                self.q_encoder.save_pretrained(self.config.path.retrieval_save_path + "/q_encoder_best_top_10")
                self.p_encoder.save_pretrained(self.config.path.retrieval_save_path + "/p_encoder_best_top_10")
            if top_25_acc > best_top_25:
                best_top_25 = top_25_acc
                self.q_encoder.save_pretrained(self.config.path.retrieval_save_path + "/q_encoder_best_top_25")
                self.p_encoder.save_pretrained(self.config.path.retrieval_save_path + "/p_encoder_best_top_25")
            if top_50_acc > best_top_50:
                best_top_50 = top_50_acc
                self.q_encoder.save_pretrained(self.config.path.retrieval_save_path + "/q_encoder_best_top_50")
                self.p_encoder.save_pretrained(self.config.path.retrieval_save_path + "/p_encoder_best_top_50")
            if top_100_acc > best_top_100:
                best_top_100 = top_100_acc
                self.q_encoder.save_pretrained(self.config.path.retrieval_save_path + "/q_encoder_best_top_100")
                self.p_encoder.save_pretrained(self.config.path.retrieval_save_path + "/p_encoder_best_top_100")
