from transformers import AutoTokenizer, RobertaForMaskedLM, ElectraForMaskedLM, BertForMaskedLM, AutoConfig, DataCollatorWithPadding, DataCollatorForLanguageModeling, AutoModelForMaskedLM
import torch
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

# fetch pretrained model for MaskedLM training 
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = AutoModelForMaskedLM.from_pretrained('klue/roberta-large')
#model = BertForMaskedLM.from_pretrained('klue/bert-base')
model = RobertaForMaskedLM.from_pretrained('klue/roberta-large')


model.to(device)

# Read txt file which is consisted of sentences from train.csv
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='../data/wikipedia_documents.txt',
    block_size=512 # block size needs to be modified to max_position_embeddings
)

data_collator = DataCollatorForLanguageModeling( 
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15 
)

# need to change arguments 
training_args = TrainingArguments(
    output_dir="./klue-roberta-retrained",
    overwrite_output_dir=True,
    learning_rate=5e-05,
    num_train_epochs=1, 
    per_device_train_batch_size=16,
    save_steps=100,
    save_total_limit=2,
    seed=42,
    save_strategy='epoch',
    gradient_accumulation_steps=8,
    logging_steps=100,
    evaluation_strategy='epoch',
    resume_from_checkpoint=True,
    fp16=True,
    fp16_opt_level='O1',
    load_best_model_at_end=True
) 

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=dataset,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
trainer.save_model("./klue-roberta-retrained")