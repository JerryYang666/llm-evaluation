# Copyright (c) 2024.
# -*-coding:utf-8 -*-
"""
@file: gsm8k_eval_single_file.py.py
@author: Jerry(Ruihuang)Yang
@email: rxy216@case.edu
@time: 9/27/24 23:45
"""
import os
import random
import json
import argparse
import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


HOME_DIR = "/home/rxy216/gsm8k"
MODEL_FILE_BASE = "/mnt/pan/courses/llm24/xxh584/"


@dataclass(frozen=True)
class Config:
    device: torch.device
    seed: int
    cache_dir: Path
    base_dir: Path


def init(seed: int = None) -> Config:
    """
    Initialize the environment settings for a machine learning project.

    Args:
        seed (int, optional): The seed for random number generators to ensure reproducibility. Defaults to None.

    Returns:
        Config: A frozen dataclass containing the configuration settings.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available")
        print("Device name:", torch.cuda.get_device_name(0))
        print("Device count:", torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print("CUDA is not available")

    # Set Hugging Face environment variables
    hf_telemetry = 1  # Set to 1 to disable telemetry
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = str(hf_telemetry)

    cache_dir = Path(f"{HOME_DIR}/.cache/misc")
    cs_home = Path(HOME_DIR)

    # Set random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return Config(device=device, seed=seed, cache_dir=cache_dir, base_dir=cs_home)


def load_gsm8k(subset="test"):
    gsm = load_dataset("openai/gsm8k", "main")

    if subset == "train":
        return gsm["train"]
    elif subset == "test":
        return gsm["test"]
    elif subset == "both":
        gsm_eval = concatenate_datasets([gsm["train"], gsm["test"]])
        gsm_eval = gsm_eval.shuffle(seed=42)
        return gsm_eval


def create_cot_prompt(
        json_file, n_shots=8, reasoning_trigger="", final_answer_trigger="", idle_trigger=""
):
    assert n_shots <= 8, "n_shots should be less than or equal to 8"

    questions = []
    chains = []
    answers = []

    with open(json_file, "r") as file:
        data = json.load(file)

    for item in data:
        questions.append(item["question"])
        chains.append(" ".join(item["reasoning"]))
        answers.append(item["answer"])

    questions, chains, answers = (
        questions[:n_shots],
        chains[:n_shots],
        answers[:n_shots],
    )

    cot_prompt = ""

    for i in range(n_shots):
        parts = []
        if reasoning_trigger:
            parts.append(reasoning_trigger)
        parts.append(chains[i])
        if final_answer_trigger:
            parts.append(final_answer_trigger)
        parts.append(answers[i])
        if idle_trigger:
            parts.append(idle_trigger)

        response = " ".join(parts)
        cot_prompt += f"Q: {questions[i]}\nA: {response}.\n\n"

    return cot_prompt


def concat_cot_prompt(cot_prompt, question):
    prompt = cot_prompt + f"Q: {question}\nA:"
    return prompt


def load_model(checkpoint):
    config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, config=config, torch_dtype=torch.float16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"
        )

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.penalty_alpha = None
    model.generation_config.do_sample = False

    model.eval()

    return model, tokenizer


def generate_response(
        prompt, model, tokenizer, generate_kwargs, remove_input_prompt=True
):
    input = tokenizer(
        prompt, padding=False, add_special_tokens=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(**input, **generate_kwargs)
    output = output[0]

    input_prompt_len = input["input_ids"].shape[1]

    if remove_input_prompt:
        output = output[input_prompt_len:]

    response = tokenizer.decode(output, skip_special_tokens=True)

    return response, input_prompt_len


def clean_response(response, final_answer_trigger="therefore"):
    response_text = " ".join(response.strip().split())

    parts = re.split(final_answer_trigger, response_text, flags=re.IGNORECASE)
    if len(parts) > 1:
        expected_text = parts[1].strip()
    else:
        parts = re.split("therefore", response_text, flags=re.IGNORECASE)
        expected_text = parts[1].strip() if len(parts) > 1 else response_text

    expected_text = expected_text.replace("$", "").replace(",", "")

    if "pm" in expected_text.lower() or "am" in expected_text.lower():
        time_match = re.search(
            r"(\d{1,2}:\d{2})\s*(am|pm)", expected_text, re.IGNORECASE
        )
        if time_match:
            time_str, meridiem = time_match.groups()
            hours, minutes = map(int, time_str.split(":"))

            if meridiem.lower() == "pm" and hours != 12:
                hours += 12
            elif meridiem.lower() == "am" and hours == 12:
                hours = 0

            time_float = hours + minutes / 60.0
            return round(time_float, 2)

    numbers = re.findall(r"-?\d+\.?\d*", expected_text)
    parsed_numbers = [float(num) if "." in num else int(num) for num in numbers]

    if parsed_numbers:
        return parsed_numbers[-1]
    else:
        exception_phrases = [
            "there is no",
            "there are no",
            "there will be no",
            "there will not be",
        ]
        if any(phrase in expected_text.lower() for phrase in exception_phrases):
            return 0

        response_before_question = response.split("Q:")[0]
        lines = response_before_question.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line:
                parts = re.split(final_answer_trigger, line, flags=re.IGNORECASE)
                expected_text = parts[1].strip() if len(parts) > 1 else line
                expected_text = expected_text.replace("$", "").replace(",", "")
                numbers = re.findall(r"-?\d+\.?\d*", expected_text)
                parsed_numbers = [
                    float(num) if "." in num else int(num) for num in numbers
                ]
                if parsed_numbers:
                    return parsed_numbers[-1]

    return None


def evaluate(
        model,
        tokenizer,
        enable_cot,
        generate_kwargs,
        n_shots=2,
        subset="test",
        iterations=None,
        save_response=None,
):
    # Triggers
    # reasoning_trigger = "Let's think step by step."
    if enable_cot:
        reasoning_trigger = "Let's break it down."
    else:
        reasoning_trigger = ""

    final_answer_trigger = "The answer is"
    # final_answer_trigger = "Therefore, the final answer is"

    # idle_trigger = "Let's go to the next question"

    cot_prompt = create_cot_prompt(
        f"{HOME_DIR}/gsm8k_rand_number.json",
        n_shots=n_shots,
        reasoning_trigger=reasoning_trigger,
        final_answer_trigger=final_answer_trigger,
        # idle_trigger=idle_trigger,
    )

    readable_responses, corrects, input_length_total, input_length_avg = 0, 0, 0, 0.0

    gsm_eval = load_gsm8k(subset=subset)
    num_questions = len(gsm_eval)

    cot_enabled = "cot_enabled" if enable_cot else "cot_disabled"

    csv_file_path_successful = f"{save_response}/successful_responses_{cot_enabled}.csv"
    csv_file_path_unsuccessful = f"{save_response}/unsuccessful_responses_{cot_enabled}.csv"

    columns = ["index", "question", "resp", "resp_parsed", "ans", "ans_parsed"]

    with open(csv_file_path_successful, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns + ["correct"])
        writer.writerow([-1, cot_prompt, "NA", "NA", "NA", "NA", "NA"])

    with open(csv_file_path_unsuccessful, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns + ["error"])
        writer.writerow([-1, cot_prompt, "NA", "NA", "NA", "NA", "NA"])

    for i in range(num_questions) if iterations is None else range(iterations):

        question, answer_truth_ = gsm_eval[i]["question"], gsm_eval[i]["answer"]
        answer_truth = answer_truth_.split(" ")[-1].replace(",", "")

        prompt = concat_cot_prompt(cot_prompt, question)

        print(f"Iteration, {i} \n")

        response, input_prompt_len = generate_response(
            prompt, model, tokenizer, generate_kwargs
        )

        # only keep the answer before the first line break
        response = response.split("\n")[0]

        final_answer_prediction = clean_response(response)

        if isinstance(final_answer_prediction, (int, float)):

            try:

                final_answer_truth = (
                    float(answer_truth) if "." in answer_truth else int(answer_truth)
                )

                if final_answer_prediction == final_answer_truth:
                    corrects += 1

                if save_response:
                    with open(csv_file_path_successful, mode="a", newline="") as file:
                        writer = csv.writer(file)

                        writer.writerow(
                            [
                                i,
                                question,
                                response,
                                final_answer_prediction,
                                answer_truth_,
                                final_answer_truth,
                                final_answer_prediction == final_answer_truth,
                            ]
                        )

                readable_responses += 1
                input_length_total += input_prompt_len

            except ValueError as e:
                print(f"Error: Value Error in iteration {i} \n")
                with open(csv_file_path_unsuccessful, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            i,
                            question,
                            response,
                            final_answer_prediction,
                            answer_truth_,
                            answer_truth,
                            f"Cannot convert true answer, {e}",
                        ]
                    )

        else:
            print(f"Error: Value Error in iteration {i} \n")
            with open(csv_file_path_unsuccessful, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        i,
                        question,
                        response,
                        final_answer_prediction,
                        answer_truth_,
                        answer_truth,
                        f"Error in parsing response",
                    ]
                )

    if readable_responses == 0:
        accuracy, input_length_avg = 0, 0

    else:
        accuracy = corrects / readable_responses
        input_length_avg = input_length_total / readable_responses

    print(f"Accuracy: {accuracy}")

    return accuracy, readable_responses, input_length_avg, num_questions


def evaluate_init(checkpoint, enable_cot, seed=None):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = Path(rf"output/GSM8K/single_{checkpoint}_{now}")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_kwargs = {
        "num_return_sequences": 1,
        "max_new_tokens": 256,
        "do_sample": False,
        # "temperature": 0.6,
        # "top_k": 50,
    }
    n_shots = 5

    checkpoint_path = MODEL_FILE_BASE + checkpoint
    model, tokenizer = load_model(checkpoint_path)

    accuracy, readable_responses, input_length_avg, num_questions = evaluate(
        model,
        tokenizer,
        enable_cot,
        generate_kwargs,
        n_shots=n_shots,
        # subset="test",
        # iterations=None,
        save_response=output_dir,
    )

    config = model.config
    config = config.to_dict()

    results = {
        "model": checkpoint,
        "accuracy": accuracy,
        "num_responses": readable_responses,
        "num_questions": num_questions,
        "num_shots": n_shots,
        "generate_kwargs": generate_kwargs,
        "config": config,
        "input_tokens_avg": input_length_avg,
        "seed": seed,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    seed = None
    system_config = init() if seed is None else init(seed=seed)
    print(system_config.device)

    checkpoints = [
        "Meta-Llama-3.1-8B-Instruct",
    ]

    # create folder and file for logging
    os.makedirs("./logs", exist_ok=True)

    logging.basicConfig(
        filename="./logs/running.log",
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    enable_cot = [False, True]

    for checkpoint in checkpoints:
        for cot in enable_cot:
            try:
                print(f"Running {checkpoint} \n")
                evaluate_init(checkpoint, cot, seed=seed)
            except Exception as e:
                logging.error(f"Error: in {checkpoint} \n {e}")
            finally:
                torch.cuda.empty_cache()
