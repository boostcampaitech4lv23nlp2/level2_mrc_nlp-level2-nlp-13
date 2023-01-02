from transformers import AutoTokenizer, AutoModelForQuestionAnswering

ckpt_path = "saved_models/klue/roberta-base/LWJ_12-27-13-36/checkpoint-14970/"
save_name = "nlpotato/roberta-base-e15"

model = AutoModelForQuestionAnswering.from_pretrained(ckpt_path)
model.push_to_hub(save_name)
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
tokenizer.push_to_hub(save_name)
