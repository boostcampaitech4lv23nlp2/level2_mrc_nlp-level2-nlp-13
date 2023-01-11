import logging
import os
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List

import omegaconf
from datasets import DatasetDict, load_metric, disable_caching
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    TrainingArguments,
)
from trainer.Seq2SeqMRCTrainer import QuestionAnsweringSeq2SeqTrainer
from utils.trainer_qa import QuestionAnsweringTrainer
from utils.utils_qa import check_sanity, postprocess_qa_predictions

logger = logging.getLogger(__name__)


@dataclass
class MRC:
    config: omegaconf.dictconfig.DictConfig
    training_args: TrainingArguments
    tokenizer: Union[AutoTokenizer, T5TokenizerFast]
    model: Union[AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, T5ForConditionalGeneration]
    datasets: Optional[DatasetDict] = None

    def __post_init__(self):
        check_sanity(self.config, self.tokenizer)
        print(self.model)
        if "QuestionAnswering" in str(type(self.model)):  # isinstance(self.model, AutoModelForQuestionAnswering):
            print("extractive")
            if self.datasets is not None:
                self.mode = "train"
                self.train_dataset = self.datasets["train"]
                self.train_dataset = self.train_dataset.map(
                    self.prepare_train_features,
                    batched=True,  # default batch_size = 1000
                    remove_columns=self.train_dataset.column_names,
                    load_from_cache_file=not self.config.utils.overwrite_cache,
                )
                self.eval_examples = self.datasets["validation"]
                self.eval_dataset = self.eval_examples.map(
                    self.prepare_validation_features,
                    batched=True,
                    remove_columns=self.eval_examples.column_names,
                    load_from_cache_file=not self.config.utils.overwrite_cache,
                )
                assert len(self.eval_dataset) != len(self.eval_examples)
            else:
                self.mode = "predict"

            # Data collator
            # flag가 True이면 이미 max length로 padding된 상태입니다.
            # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
            data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8 if self.config.train.fp16 else None)

            def model_init():
                return AutoModelForQuestionAnswering.from_pretrained(
                    self.config.model.name_or_path,
                    from_tf=bool(".ckpt" in self.config.model.name_or_path),
                )

            # Trainer 초기화
            if self.mode == "train":
                print("initialize QuestionAnsweringTrainer")
                self.trainer = QuestionAnsweringTrainer(
                    model_init=model_init,
                    model=self.model,
                    args=self.training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                    eval_examples=self.eval_examples,
                    tokenizer=self.tokenizer,
                    data_collator=data_collator,
                    post_process_function=self.post_processing_function,
                    compute_metrics=self.compute_metrics,
                )

            else:
                # inference
                self.trainer = QuestionAnsweringTrainer(
                    model=self.model,
                    args=self.training_args,
                    tokenizer=self.tokenizer,
                    data_collator=data_collator,
                    post_process_function=self.post_processing_function,
                    compute_metrics=self.compute_metrics,
                )
        elif isinstance(self.model, AutoModelForSeq2SeqLM) or isinstance(self.model, T5ForConditionalGeneration):
            print(" *** generative *** ")
            if self.datasets is not None:
                self.mode = "train"
                self.train_dataset = self.datasets["train"]
                self.train_dataset = self.train_dataset.map(
                    self.preprocess_train_features_generative,
                    batched=True,  # default batch_size = 1000
                    remove_columns=self.train_dataset.column_names,
                    load_from_cache_file=not self.config.utils.overwrite_cache,
                )
                self.eval_examples = self.datasets["validation"]
                self.eval_dataset = self.eval_examples.map(
                    self.preprocess_validation_features_generative,
                    batched=True,
                    remove_columns=self.eval_examples.column_names,
                    load_from_cache_file=not self.config.utils.overwrite_cache,
                )
                assert len(self.eval_dataset) != len(self.eval_examples)
            else:
                self.mode = "predict"

            data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, label_pad_token_id=self.tokenizer.pad_token_id)
            gen_config = dict(self.config.get("generation"))
            if self.mode == "train":
                self.trainer = QuestionAnsweringSeq2SeqTrainer(
                    model=self.model,
                    args=self.training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                    eval_examples=self.eval_examples,
                    tokenizer=self.tokenizer,
                    data_collator=data_collator,
                    compute_metrics=self.compute_metrics,
                    post_process_function=self.post_processing_function_generative,
                    gen_config=gen_config,
                )
            else:
                self.trainer = QuestionAnsweringSeq2SeqTrainer(
                    model=self.model,
                    args=self.training_args,
                    tokenizer=self.tokenizer,
                    data_collator=data_collator,
                    compute_metrics=self.compute_metrics,
                    post_process_function=self.post_processing_function_generative,
                    gen_config=gen_config,
                )

    def train(self, checkpoint=None):
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(self.train_dataset)

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        output_train_file = os.path.join(self.training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"{key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        self.trainer.state.save_to_json(os.path.join(self.training_args.output_dir, "trainer_state.json"))

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        logger.info("*** Evaluate ***")
        
        metrics = self.trainer.evaluate(
            eval_dataset=eval_dataset,  # prepocessed
            eval_examples=eval_examples,  # unpreprocessed
            ignore_keys=ignore_keys,
        )
        metrics["eval_samples"] = len(self.eval_dataset)

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def predict(self, predict_dataset, ignore_keys=None):
        logger.info("*** Predict ***")
        self.mode = "predict"
        self.predict_dataset = predict_dataset
        self.predict_dataset = self.predict_dataset.map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=predict_dataset.column_names,
            load_from_cache_file=not self.config.utils.overwrite_cache,
        )
        self.trainer.predict(
            predict_dataset=self.predict_dataset,
            predict_examples=predict_dataset,
            ignore_keys=ignore_keys,
        )

    def prepare_train_features(self, examples):
        """
        context와 query를 더했을 때 너무 길면 context를 잘라서 활용.
        잘라진 context에서 정답이 있는 위치를 파악하는데, 해당 context에 없으면 cls의 인덱스를 리턴.
        """
        pad_on_right = self.tokenizer.padding_side == "right"
        column_names = self.train_dataset.column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        tokenized_examples = self.tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            **self.config.tokenizer,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)  # cls index
            sequence_ids = tokenized_examples.sequence_ids(i)  # 0 for seq1 tokens, 1 for seq2 tokens

            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # true Start/end character indices
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self, examples):
        pad_on_right = self.tokenizer.padding_side == "right"
        if self.mode == "train":
            column_names = self.eval_examples.column_names
        elif self.mode == "predict":
            column_names = self.predict_dataset.column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]

        tokenized_examples = self.tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            **self.config.tokenizer,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (offsets if sequence_ids[k] == context_index else None) for k, offsets in enumerate(tokenized_examples["offset_mapping"][i])
            ]

            print(f"id: {examples['id'][sample_index]}, inputs: {self.tokenizer.decode(tokenized_examples['input_ids'][i])}")
            if i == 10:
                break
        return tokenized_examples

    def post_processing_function(self, examples, features, predictions):
        """
        Evaluation/Prediction에서 start logits과 end logits을 original context의 정답과 match하는 함수
        """
        if self.config.retriever.type == "sparse":
            args = self.config.sparse
            prefix = f"tfidf{args.tfidf_num_features}"
            if args.lsa:
                prefix += f"_lsa{args.lsa_num_features}"
            if self.config.faiss.use_faiss:
                prefix += f"_faiss{self.config.faiss.num_clusters}_{self.config.faiss.metric}"

            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                max_answer_length=self.config.utils.max_answer_length,
                output_dir=self.training_args.output_dir,
                prefix=prefix,
            )
        elif self.config.retriever.type == "dense" or "hybrid":
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                max_answer_length=self.config.utils.max_answer_length,
                output_dir=self.training_args.output_dir,
                prefix=None,
            )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        if self.mode == "train":
            column_names = examples.column_names
            answer_column_name = "answers" if "answers" in column_names else column_names[2]
            references = [{"id": example["id"], "answers": example[answer_column_name]} for example in examples]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

        elif self.mode == "predict":
            return formatted_predictions

    def compute_metrics(self, p: EvalPrediction):
        metric = load_metric("squad")
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    def preprocess_squad_batch(
        self,
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
    ) -> Tuple[List[str], List[str]]:
        questions = examples[question_column]
        contexts = examples[context_column]
        answers = examples[answer_column]

        def generate_input(_question, _context):
            return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

        inputs = [generate_input(question, context) for question, context in zip(questions, contexts)]
        targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
        return inputs, targets

    def preprocess_train_features_generative(self, examples):
        column_names = self.train_dataset.column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]
        inputs, targets = self.preprocess_squad_batch(examples, question_column_name, context_column_name, answer_column_name)

        inputs = self.tokenizer(
            inputs,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            **self.config.tokenizer,
        )

        # Tokenize targets with text_target=...
        labels = self.tokenizer(
            text_target=targets,
            truncation=True,
            **self.config.tokenizer,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.config.tokenizer.padding == "max_length":  # and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs.pop("offset_mapping")

        new_input_ids, new_attention_mask, new_token_type_ids = [], [], []
        new_sample_mapping = []
        for i, offsets in enumerate(offset_mapping):
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            start_char_idx = answers["answer_start"][0]
            end_char_idx = start_char_idx + len(answers["text"][0])
            valid_max_idx = len(offsets) - 1
            while offsets[valid_max_idx] == (0, 0):
                # find the offset tuple of the last non-padding_token
                valid_max_idx -= 1
            if start_char_idx >= offsets[1][0] and end_char_idx < offsets[valid_max_idx][1]:
                # if the answer is in the truncated input string
                # start_char_idx is compared to offsets[1][0], since offsets[0][0] corresponds to a special token like [CLS]
                # likewise, end_char_idx is compared to offsets[-2][0], since the last offset is a special token
                new_input_ids.append(inputs["input_ids"][i])
                new_attention_mask.append(inputs["attention_mask"][i])
                new_token_type_ids.append(inputs["token_type_ids"][i])
                new_sample_mapping.append(sample_mapping[i])

        inputs["input_ids"] = new_input_ids
        inputs["attention_mask"] = new_attention_mask
        inputs["token_type_ids"] = new_token_type_ids

        # Augment the overflowing tokens to the labels
        labels_out = []

        for i in range(len(inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = new_sample_mapping[i]
            labels_out.append(labels["input_ids"][sample_index])

        inputs["labels"] = labels_out
        assert len(inputs["input_ids"]) == len(inputs["labels"])
        return inputs

    # Validation preprocessing
    def preprocess_validation_features_generative(self, examples):
        # column_names = self.eval_dataset.column_names
        question_column_name = "question"  # if "question" in column_names else column_names[0]
        context_column_name = "context"  # if "context" in column_names else column_names[1]
        answer_column_name = "answers"  # if "answers" in column_names else column_names[2]
        inputs, targets = self.preprocess_squad_batch(examples, question_column_name, context_column_name, answer_column_name)

        model_inputs = self.tokenizer(
            inputs,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            **self.config.tokenizer,
        )
        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(
            text_target=targets,
            truncation=True,
            **self.config.tokenizer,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # # padding in the loss.
        if self.config.tokenizer.padding == "max_length":  # and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        model_inputs["example_id"] = []
        # Augment the overflowing tokens to the labels
        labels_out = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])
            labels_out.append(labels["input_ids"][sample_index])
        model_inputs["labels"] = labels_out
        return model_inputs

    def post_processing_function_generative(self, examples, features, outputs, stage="eval"):
        # decode
        print("post_processing")
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # build a map example to its corresponding features
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}

        for example_index, example in enumerate(examples):
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        # import pandas as pd
        import json

        # df = pd.DataFrame(formatted_predictions)
        # df.to_csv("t5_predictions_new.csv")
        # references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        # return EvalPrediction(predictions=formatted_predictions, label_ids=references)

        if self.mode == "train":
            column_names = examples.column_names
            answer_column_name = "answers" if "answers" in column_names else column_names[2]
            references = [{"id": example["id"], "answers": example[answer_column_name]} for example in examples]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

        elif self.mode == "predict":
            output_dir = self.config.train.output_dir
            if not os.path.isdir(output_dir):
                os.makedir(output_dir)

            prediction_file = os.path.join(
                output_dir,
                "predictions_t5.json",
            )
            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w", encoding="utf-8") as writer:
                print("Saving json")
                writer.write(json.dumps(predictions, indent=4, ensure_ascii=False) + "\n")
        return formatted_predictions
