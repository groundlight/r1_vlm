import json
import os
import random

# chooses examples from the training set that are good candidates for training with tool use.


def find_examples_for_training(
    eval_results_file_path: str = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/tool_use_text_vqa_env/eval_on_train_results.jsonl",
):
    """
    We want to select examples from the train set where the model is performing poorly without using tools.

    Args:
        eval_results_file_path: path to the file containing base model's evaluation results on the train set.
    Returns:
        list of question_ids that are good candidates for training with tool use.
    """

    results = []
    if os.path.exists(eval_results_file_path):
        with open(eval_results_file_path, "r") as f:
            results = [json.loads(line) for line in f]
    else:
        raise FileNotFoundError(f"File {eval_results_file_path} not found")

    print(f"There are {len(results)} results in the file {eval_results_file_path}")

    # only keep examples with vqa score == 0
    results = [result for result in results if result["vqa_score"] == 0]

    question_ids = [result["question_id"] for result in results]

    print(
        f"There are {len(question_ids)} question_ids with vqa score == 0 that we can use for training"
    )

    return question_ids


def find_examples_for_training_1_to_1_mix(
    eval_results_file_path: str = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/tool_use_text_vqa_env/eval_on_train_results.jsonl",
):
    """
    Returns a dataset that is a 1:1 mix of easy to hard examples, where
    easy == vqa score > 0
    hard == vqa score == 0
    """

    results = []
    if os.path.exists(eval_results_file_path):
        with open(eval_results_file_path, "r") as f:
            results = [json.loads(line) for line in f]
    else:
        raise FileNotFoundError(f"File {eval_results_file_path} not found")

    print(f"There are {len(results)} results in the file {eval_results_file_path}")

    easy_results = [result for result in results if result["vqa_score"] > 0]
    hard_results = [result for result in results if result["vqa_score"] == 0]

    # there should be more easy than hard

    if not len(easy_results) > len(hard_results):
        raise ValueError(
            f"There should be more easy than hard results, but there are {len(easy_results)} easy and {len(hard_results)} hard"
        )

    # take all the hard results, and then sample without replacement from the easy results until we have the same number of easy and hard results
    sampled_easy_results = random.sample(easy_results, len(hard_results))

    # combine the results and shuffle
    combined_results = hard_results + sampled_easy_results
    random.shuffle(combined_results)

    # now take their question ids
    question_ids = [result["question_id"] for result in combined_results]

    return question_ids


if __name__ == "__main__":
    find_examples_for_training()

    find_examples_for_training_1_to_1_mix()
