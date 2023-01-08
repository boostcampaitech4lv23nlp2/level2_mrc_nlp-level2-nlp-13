import json
import os
import random
import warnings

import torch
from pororo import Pororo
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments

warnings.filterwarnings("ignore")


class MLMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SalientSpanMaskingCollator(DefaultDataCollator):
    def __init__(self, tokenizer):
        self.mask_token_id = tokenizer.mask_token_id  # 4
        self.max_length = 256
        self.ner = Pororo(task="ner", lang="ko")

    def __call__(self, examples):
        # encode_plus()로 input_ids, attention_mask, token_type_ids를 구함
        tokenized_examples = tokenizer.batch_encode_plus(
            examples, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True, add_special_tokens=False
        )
        attention_mask = tokenized_examples["attention_mask"]
        origin_input_ids = tokenized_examples["input_ids"]
        masking_patterns = [self.salient_span_masking(example, self.mask_token_id)["input_ids"] for example in examples]

        masking_patterns = torch.tensor(masking_patterns, dtype=torch.long)

        # input_ids와 masking_patterns를 비교하여 mask_token_id로 바꿔줌
        input_ids = origin_input_ids.clone()
        input_ids[masking_patterns == self.mask_token_id] = self.mask_token_id

        # input_ids에서 mask가 아닌 부분은 -100으로, mask는 원래 값으로 바꿔줌
        labels = origin_input_ids.clone()
        labels[masking_patterns != self.mask_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def salient_span_masking(self, input_str, mask_token_id):

        ner_output = self.ner(input_str)

        # ner인 token을 리스트에 추가
        ner_words = []
        for i, w in enumerate(ner_output):
            if w[1] != "O":
                ner_words.append(w[0])

        # ner word가 있다면 해당 word의 span을 찾기 # 없을 경우는 랜덤으로 하나 선택
        if len(ner_words) == 0:
            ner_count = len(ner_output)
            while True:
                random_idx = random.randint(0, ner_count - 1)
                if ner_output[random_idx][0] != " ":
                    ner_words.append(ner_output[random_idx][0])
                    break

        ner_word_id_spans = []
        for ner_word in ner_words:
            temp = tokenizer(ner_word, add_special_tokens=False)
            ner_word_id_span = temp["input_ids"]
            ner_word_id_spans.append(ner_word_id_span)

        # ner word span을 mask token으로 바꾸기
        tokenized_sent = tokenizer(input_str, add_special_tokens=False, max_length=self.max_length, padding="max_length", truncation=True)

        if len(ner_word_id_spans) > 5:
            ner_word_id_spans = random.sample(ner_word_id_spans, 5)

        for ner_word_id_span in ner_word_id_spans:
            for i, id in enumerate(tokenized_sent["input_ids"]):
                try:
                    if id == ner_word_id_span[0] and tokenized_sent["input_ids"][i + len(ner_word_id_span) - 1] == ner_word_id_span[-1]:
                        tokenized_sent["input_ids"][i : i + len(ner_word_id_span)] = [mask_token_id] * len(ner_word_id_span)
                        break
                except:
                    pass

        return tokenized_sent


if __name__ == "__main__":
    model_name = "klue/roberta-large"

    # Pretrained tokenizer & Model for MaskedLM training
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    mask_token = tokenizer.mask_token_id  # 4

    if os.path.isfile("./data/pretrain/wikipedia_documents.txt"):
        print("File already exists")
    else:
        os.mkdir("./data/pretrain")
        with open("./data/wikipedia_documents.json", "r") as f:
            wiki = json.load(f)

        with open("./data/pretrain/wikipedia_documents.txt", "w") as f:
            wiki = list(dict.fromkeys(w["text"] for w in wiki.values()))
            for w in wiki:
                f.write(w)

    with open("./data/pretrain/wikipedia_documents.txt", "r") as f:
        data = []
        for line in f:
            if len(line) <= 5:
                pass
            else:
                data.append(line.rstrip("\n"))

    # Dataset
    dataset = MLMDataset(data)
    data_collator = SalientSpanMaskingCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./pretrained_model",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        save_steps=4000,
        save_total_limit=3,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("./pretrained_model")
