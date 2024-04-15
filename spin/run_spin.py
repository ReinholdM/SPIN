#!/usr/bin/env python
# 
# Adapted from https://github.com/huggingface/alignment-handbook 
import logging
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from accelerate import Accelerator
from alignment import (
    DataArguments,
    SPINConfig,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from alignment import SPINTrainer
from torch.utils.data import Subset
import re

from datasets import load_dataset

def apply_chat_template(
    example, tokenizer, task, assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if all(k in example.keys() for k in ("real", "generated")):
        # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
        prompt_messages = [[msg for msg in example["real"] if msg["role"] == "user"][0]]
        # Insert system message
        if example["real"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            prompt_messages.insert(0, example["real"][0])

        real_messages = example["real"][1:]
        generated_messages = example["generated"][1:]
        example["text_real"] = tokenizer.apply_chat_template(real_messages, tokenize=False)
        example["text_generated"] = tokenizer.apply_chat_template(generated_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        example["text_real"] = _strip_prefix(example["text_real"], assistant_prefix)
        example["text_generated"] = _strip_prefix(example["text_generated"], assistant_prefix)
    else:
        raise ValueError(
            f"Require `[real, generated]` keys but found {list(example.keys())}"
            )
    return example

logger = logging.getLogger(__name__)

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""



def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SPINConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
        mask = False
        if mask:
            print('Masking Obersevation')
            tokenizer.mask_token = "~"

            # Split the input text into lines
            lines = prompt.split('\n')

            # Initialize an empty list to store the modified lines
            masked_lines = []

            # Initialize a flag to indicate if we are between "Observation:" and "Thought:"
            between_observation_and_thought = False

            # Iterate through each line
            for line in lines:
                if "Observation:" in line:
                    between_observation_and_thought = True
                    #split the line and mask all but the first word
                    line = line.split()
                    line[1:] = [tokenizer.mask_token] * len(line[1:])
                    line = " ".join(line)
                    masked_lines.append(line)  # Add the line as-is
                else:
                    masked_lines.append(line)  # Add the line as-is

            # Concatenate the modified lines to form the masked text
            masked_text = '\n'.join(masked_lines)

            prompt = masked_text

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=training_args.max_length,
            padding=False,
            return_tensors=None
        )

        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < training_args.max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        else:
            if len(result["input_ids"]) >= training_args.max_length:
                print("WARNING: input too long, truncating")

        masked_token_id = tokenizer.mask_token_id
        ids = [-100 if token_id == 3695 else token_id for token_id in result["input_ids"]]

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        train_on_inputs = True
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    #####################
    # Apply chat template
    #####################
    # raw_datasets = raw_datasets.map(
    #     apply_chat_template,
    #     fn_kwargs={"tokenizer": tokenizer, "task": "spin"},
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     desc="Formatting comparisons with prompt template",
    # )

    # Replace column names with what TRL needs, text_real -> real and text_generated -> generated
    # for split in ["train", "test"]:
    #     raw_datasets[split] = raw_datasets[split].rename_columns(
    #         {"text_prompt": "prompt", "text_real": "real", "text_generated": "generated"}
    #     )

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        load_in_8bit=True,
    )

    model = model_args.model_name_or_path

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    ###############
    # Load datasets
    ###############
    data_path = "data/finetune/alpaca_format/hotpotqa.json"
    data = load_dataset("json", data_files=data_path)
    
    
    train_val = data["train"].train_test_split(
        test_size=0.01, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    
    # raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    # logger.info(
    #     f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    # )
    # column_names = list(raw_datasets["train"].features)
    
    #########################
    # Instantiate spin trainer
    #########################
    spin_trainer = SPINTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            warmup_steps=100,
            num_train_epochs=30,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200 ,
            save_steps=200,
            output_dir=training_args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=True,
        ),
        beta=training_args.beta,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    train_result = spin_trainer.train()
    # metrics = train_result.metrics
    # max_train_samples = (
    #     data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    # )
    # metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    # spin_trainer.log_metrics("train", metrics)
    # spin_trainer.save_metrics("train", metrics)
    # spin_trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    spin_trainer.save_model(training_args.output_dir)
    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        spin_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        spin_trainer.model.config.use_cache = True
        spin_trainer.model.config.save_pretrained(training_args.output_dir)

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
