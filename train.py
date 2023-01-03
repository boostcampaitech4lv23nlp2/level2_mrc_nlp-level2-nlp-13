import logging
import os
import sys
import argparse
import wandb
import pytz
import datetime

from mrc import MRC
from omegaconf import OmegaConf
from datasets import load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


def main(args):
    config = OmegaConf.load(f"./config/{args.config}.yaml")
    # wandb 설정
    now_time = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")
    run_id = f"MRC_{config.wandb.name}_{now_time}"
    wandb.init(
        entity=config.wandb.team,
        project=config.wandb.project,
        group=config.model.name_or_path,
        id=run_id,
        tags=config.wandb.tags,
        config=config,
    )
    # update config for a sweep run
    if args.learning_rate is not None:
        config.optimizer.learning_rate = args.learning_rate
    if args.epoch is not None:
        config.train.num_train_epochs = args.epoch
    if args.batch_size is not None:
        config.train.per_device_train_batch_size = args.batch_size

    config.train.update(config.optimizer)
    if config.train.output_dir is None:
        config.train.output_dir = os.path.join("saved_models", config.model.name_or_path, run_id)
    training_args = TrainingArguments(**config.train)
    training_args.report_to = ["wandb"]

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("config", config)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(config.utils.seed)

    datasets = load_from_disk(config.path.train)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        from_tf=bool(".ckpt" in config.model.name_or_path),
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        config.model.name_or_path,
        from_tf=bool(".ckpt" in config.model.name_or_path),
    )

    reader = MRC(
        config=config,
        training_args=training_args,
        tokenizer=tokenizer,
        model=model,
        datasets=datasets,
    )
    reader.train(checkpoint=config.path.resume)
    eval_metrics = reader.evaluate()
    wandb.log(
        {
            "eval/exact_match": eval_metrics["exact_match"],
            "eval/f1": eval_metrics["f1"],
        }
    )

    # share the pretrained model to huggingface hub
    if config.hf_hub.push_to_hub is True:
        save_name = config.hf_hub.save_name
        if not save_name.startswith("nlpotato/"):
            save_name = "nlpotato/" + save_name
        model.push_to_hub(config.hf_hub.save_name)
        tokenizer.push_to_hub(config.hf_hub.save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="custom_config")
    parser.add_argument("--learning_rate", "-lr", type=float, required=False)
    parser.add_argument("--epoch", type=int, required=False)
    parser.add_argument("--batch_size", "-bs", type=int, required=False)
    parser.add_argument("--weight_decay", "-wd", required=False)
    args, _ = parser.parse_known_args()
    main(args)
