import json
import logging
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DenseRetrievalTrainer:
    def __init__(self, args, config, tokenizer, p_encoder, q_encoder, train_dataset, valid_dataset):
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        with open("./data/wikipedia_documents.json", "r") as f:
            wiki = json.load(f)
        self.wiki_contexts = list(dict.fromkeys(w["text"] for w in wiki.values()))

    def configure_optimizers(self, optimizer_grouped_parameters, config):
        if config.DPR.optimizer.name == "AdamW":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.DPR.optimizer.learning_rate)
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

            # hn_inputs = {
            #     "input_ids": batch[6],  # (batch_size, 256)
            #     "attention_mask": batch[7],  # (batch_size, 256)
            #     "token_type_ids": batch[8],  # (batch_size, 256)
            # }

            if self.config.DPR.train.hard_negative:  # hard negative 설정
                q_outputs = self.q_encoder(**q_inputs)  # (batch_size, 768)

                # Calculate Similarity score & loss
                """
                hard negatvie 연산
                query (batch_size, 768) * passage^T (768, batch_size + 1)
                """
                sim_score = torch.cuda.FloatTensor()
                for i in range(len(batch[0])):
                    p_input = {
                        "input_ids": torch.cat([p_inputs["input_ids"], torch.unsqueeze(batch[6][i], dim=0)], dim=0),
                        "attention_mask": torch.cat([p_inputs["attention_mask"], torch.unsqueeze(batch[7][i], dim=0)], dim=0),
                        "token_type_ids": torch.cat([p_inputs["token_type_ids"], torch.unsqueeze(batch[8][i], dim=0)], dim=0),
                    }
                    p_output = self.p_encoder(**p_input)

                    sim_score = torch.cat([torch.unsqueeze(torch.matmul(q_outputs[i], torch.transpose(p_output, 0, 1)), dim=1), sim_score])
                sim_score = sim_score.reshape(q_outputs.size()[0], -1)  # (batch_size, batch_size + 1)
            else:
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

    def valid_per_epoch(self, valid_dataloader: DataLoader, epoch):
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

        if not os.path.exists("./saved_models/DPR/passage_embedding_vectors/"):
            os.makedirs("./saved_models/DPR/passage_embedding_vectors/")
        with open(f"./saved_models/DPR/passage_embedding_vectors/p_embs_epoch_{epoch}.bin", "wb") as f:
            pickle.dump(p_embs, f)

        # Question Embeddig 생성
        top_10, top_20, top_30 = 0, 0, 0

        valid_data = load_from_disk("./data/train_dataset/validation")
        valid_question = valid_data["question"]
        valid_context = valid_data["context"]

        # validation 결과 데이터프레임 생성
        checking_df = pd.DataFrame()

        with torch.no_grad():
            self.q_encoder.eval()
            for idx in tqdm(range(len(valid_question))):
                query = valid_question[idx]

                q_seq = self.tokenizer(
                    [query],
                    max_length=self.config.DPR.tokenizer.max_question_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to("cuda")
                q_emb = self.q_encoder(**q_seq).to("cpu")

                # Cosine Similarity 계산
                dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
                rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()  # (1,num_passage) -> (num_passage)

                top_10_passages = [self.wiki_contexts[i] for i in rank[:10]]
                top_20_passages = [self.wiki_contexts[i] for i in rank[:20]]
                top_30_passages = [self.wiki_contexts[i] for i in rank[:30]]

                # top_k Accuracy 계산
                if valid_context[idx] in top_10_passages:
                    top_10 += 1
                    checking_df = checking_df.append(
                        {
                            "question": query,
                            "answer_context": valid_context[idx],
                            "top_10": top_10_passages,
                            "is_in": 1,
                        },
                        ignore_index=True,
                    )
                if valid_context[idx] not in top_10_passages:
                    checking_df = checking_df.append(
                        {
                            "question": query,
                            "answer_context": valid_context[idx],
                            "top_10": top_10_passages,
                            "is_in": 0,
                        },
                        ignore_index=True,
                    )
                if valid_context[idx] in top_20_passages:
                    top_20 += 1
                if valid_context[idx] in top_30_passages:
                    top_30 += 1

        if self.config.DPR.utils.valid_analysis:
            if not os.path.exists("./results/DPR/"):
                os.makedirs("./results/DPR/")
            checking_df.to_csv(f"./results/DPR/checking_df_epoch_{epoch}.csv", index=False)
        return (
            top_10 / len(valid_question) * 100,
            top_20 / len(valid_question) * 100,
            top_30 / len(valid_question) * 100,
        )

    def train(self):
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  batch size  = %d", self.args.per_device_train_batch_size)

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.per_device_train_batch_size, drop_last=True)
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.args.per_device_eval_batch_size)

        best_top_10, best_top_20, best_top_30 = 0, 0, 0

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
            top_10_acc, top_20_acc, top_30_acc = self.valid_per_epoch(valid_dataloader, epoch)

            logger.info("***** Validation Result *****")
            logger.info(
                f"epoch: {epoch} | train loss: {train_loss:.4f} | top_10_acc: {top_10_acc:.2f} | top_20_acc: {top_20_acc:.2f} | top_30_acc: {top_30_acc:.2f} "
            )
            print(
                f"epoch: {epoch} | train loss: {train_loss:.4f} | top_10_acc: {top_10_acc:.2f} | top_20_acc: {top_20_acc:.2f} | top_30_acc: {top_30_acc:.2f} "
            )

            scheduler.step()

            # Save model 나중에 validation해서 최고 기록 낸 모델 저장으로 수정
            if top_10_acc > best_top_10:
                best_top_10 = top_10_acc
                self.q_encoder.save_pretrained("./saved_models/DPR/encoder/q_encoder_best_top_10")
                self.p_encoder.save_pretrained("./saved_models/DPR/encoder/p_encoder_best_top_10")
            if top_20_acc > best_top_20:
                best_top_20 = top_20_acc
                self.q_encoder.save_pretrained("./saved_models/DPR/encoder/q_encoder_best_top_20")
                self.p_encoder.save_pretrained("./saved_models/DPR/encoder/p_encoder_best_top_20")
            if top_30_acc > best_top_30:
                best_top_30 = top_30_acc
                self.q_encoder.save_pretrained("./saved_models/DPR/encoder/q_encoder_best_top_30")
                self.p_encoder.save_pretrained("./saved_models/DPR/encoder/p_encoder_best_top_30")
