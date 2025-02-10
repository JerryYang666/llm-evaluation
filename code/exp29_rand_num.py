# Copyright (c) 2024.
# -*-coding:utf-8 -*-
"""
@file: exp28_t2.py
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

HOME_DIR = "/home/rxy216/exp29"
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
    # only keep the answer before the first line break
    response = response.split("\n")[0]

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


def run_evaluations(
        model,
        tokenizer,
        prompt_iterable,
        n_shots_iterable,
        response_cleaner,
        generate_kwargs,
        response_save_dir,
        n_shots=2,
        iterations=None,
        experiment_tag="",
):
    """
    Run evaluations for a list of prompts and n_shots.
    :param model: The path to the model checkpoint we want to evaluate.
    :param tokenizer: The tokenizer for the model.
    :param prompt_iterable: An iterable of questions to evaluate. Each question should be a tuple of the question and
    the metadata for the question.
    :param n_shots_iterable: An iterable of n_shots (in-context learning shots) to for the evaluation.
    :param response_cleaner: The function to clean the response. This function should take the response and return the
    cleaned response.
    :param generate_kwargs: The generation kwargs for the model.
    :param response_save_dir: Where to save the responses.
    :param n_shots: How many shots to use for the evaluation.
    :param iterations: How many prompts to evaluate.
    :param experiment_tag: A tag to add to the response file names.
    :return:
    """
    readable_responses, corrects, input_length_total, input_length_avg, num_questions = 0, 0, 0, 0.0, 0

    csv_file_path_successful = f"{response_save_dir}/successful_responses_{experiment_tag}.csv"
    csv_file_path_unsuccessful = f"{response_save_dir}/unsuccessful_responses_{experiment_tag}.csv"

    columns = ["index", "question", "resp", "resp_parsed", "ans", "ans_parsed", "question_metadata"]

    cot_prompt = [n_shots_iterable[prompt_no] for prompt_no in range(n_shots)]

    with open(csv_file_path_successful, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns + ["correct"])
        writer.writerow([-1, cot_prompt, "NA", "NA", "NA", "NA", "NA"])

    with open(csv_file_path_unsuccessful, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns + ["error"])
        writer.writerow([-1, cot_prompt, "NA", "NA", "NA", "NA", "NA"])

    # generate the shots based on n_shots
    n_shot_counter = 0
    n_shots_prompts = []
    for shot in n_shots_iterable:
        if n_shots is not None and n_shot_counter > n_shots:
            break
        n_shots_prompts.append(shot)
        n_shot_counter += 1
    n_shots_prompt = "\n".join(n_shots_prompts) + "\n"

    # start the evaluation
    eval_counter = 0
    for question, question_metadata, answer_truth in prompt_iterable:
        if iterations is not None and eval_counter > iterations:
            break

        print(f"Iteration, {eval_counter} \n")

        response, input_prompt_len = generate_response(
            n_shots_prompt + question, model, tokenizer, generate_kwargs
        )

        final_answer_prediction = response_cleaner(response)

        if isinstance(final_answer_prediction, (int, float)):

            try:
                print(f"Question: {answer_truth}")
                final_answer_truth = int(answer_truth)

                if final_answer_prediction == final_answer_truth:
                    corrects += 1

                with open(csv_file_path_successful, mode="a", newline="") as file:
                    writer = csv.writer(file)

                    writer.writerow(
                        [
                            eval_counter,
                            question,
                            response,
                            final_answer_prediction,
                            answer_truth,
                            final_answer_truth,
                            question_metadata,
                            final_answer_prediction == final_answer_truth,
                        ]
                    )

                readable_responses += 1
                input_length_total += input_prompt_len

            except ValueError as e:
                print(f"Error: Value Error in iteration {eval_counter} \n")
                with open(csv_file_path_unsuccessful, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            eval_counter,
                            question,
                            response,
                            final_answer_prediction,
                            answer_truth,
                            answer_truth,
                            f"Cannot convert true answer, {e}",
                        ]
                    )

        else:
            print(f"Error: Value Error in iteration {eval_counter} \n")
            with open(csv_file_path_unsuccessful, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        eval_counter,
                        question,
                        response,
                        final_answer_prediction,
                        answer_truth,
                        answer_truth,
                        f"Error in parsing response",
                    ]
                )
        eval_counter += 1

    if readable_responses == 0:
        accuracy, input_length_avg = 0, 0

    else:
        accuracy = corrects / readable_responses
        input_length_avg = input_length_total / readable_responses

    print(f"Accuracy: {accuracy}")

    return accuracy, readable_responses, input_length_avg, num_questions


def prompt_iterable_func():
    """
    generate the prompt iterable, format: (prompt, metadata, answer)
    :return:
    """
    prompt_iterable = []
    random_number_prompts = ["Pick a random number between 1 and 100. The number is: ",
                             "From 1 to 100, select a random number. You chose: ",
                             "You have 100 apples numbered from 1 to 100. Choose one. You picked apple number: "]
    for i in range(1, 12001):
        for j in range(0, 3):
            prompt_iterable.append((random_number_prompts[j], j, 0))
    return prompt_iterable


def n_shots_iterable_func():
    """
    generate the n_shots iterable
    :return:
    """
    n_shots_iterable = []
    for _ in range(50):
        i = random.randint(1, 10)
        j = random.randint(1, 10)
        i_digits = [random.randint(0, 9) for _ in range(i)]
        i_number = int(''.join(map(str, i_digits)))
        j_digits = [random.randint(0, 9) for _ in range(j)]
        j_number = int(''.join(map(str, j_digits)))
        n_shots_iterable.append(f"What is {i_number} * {j_number}? The answer is: {i_number * j_number}")
    return n_shots_iterable


def evaluate_init(checkpoint, seed=None):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = Path(rf"output/exp28/single_{checkpoint}_{now}")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_kwargs = {
        "num_return_sequences": 1,
        "max_new_tokens": 8,
        "do_sample": True,
        # "temperature": 0.6,
        # "top_k": 50,
    }
    n_shots = 0

    checkpoint_path = MODEL_FILE_BASE + checkpoint
    model, tokenizer = load_model(checkpoint_path)

    prompt_iterable = prompt_iterable_func()
    n_shots_iterable = n_shots_iterable_func()

    accuracy, readable_responses, input_length_avg, num_questions = run_evaluations(
        model,
        tokenizer,
        prompt_iterable,
        n_shots_iterable,
        clean_response,
        generate_kwargs,
        output_dir,
        n_shots=n_shots,
        experiment_tag="rand_number",
        # iterations=None,
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
        "Llama-3.2-1B",
    ]

    # create folder and file for logging
    os.makedirs("./logs", exist_ok=True)

    logging.basicConfig(
        filename="./logs/running.log",
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s [%(pathname)s:%(lineno)d]",
    )

    for checkpoint in checkpoints:
        try:
            print(f"Running {checkpoint} \n")
            evaluate_init(checkpoint, seed=seed)
        finally:
            torch.cuda.empty_cache()
