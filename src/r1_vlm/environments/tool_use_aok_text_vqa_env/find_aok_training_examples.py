import json
import os


def find_AOK_training_examples_less_than_0_5_pass_rate(
    eval_results_file_path: str = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/tool_use_aok_text_vqa_env/eval_on_aok_train_results_8shot.jsonl",
):
    """
    Returns a dataset of question_ids where the average score on the base model 8 shot is less than or equal to 0.5
    """

    results = []
    if os.path.exists(eval_results_file_path):
        with open(eval_results_file_path, "r") as f:
            results = [json.loads(line) for line in f]
    else:
        raise FileNotFoundError(f"File {eval_results_file_path} not found")

    # group the results by question_id
    results_by_question_id = {}
    for result in results:
        if result["question_id"] not in results_by_question_id:
            results_by_question_id[result["question_id"]] = {
                "results": [],
            }
        results_by_question_id[result["question_id"]]["results"].append(result)

    # compute the average score for each question_id
    for question_id in results_by_question_id:
        results = results_by_question_id[question_id]["results"]
        average_score = sum([result["score"] for result in results]) / len(results)
        results_by_question_id[question_id]["average_score"] = average_score

    # choose the examples with average score <= 0.5
    question_ids = [
        question_id
        for question_id in results_by_question_id
        if results_by_question_id[question_id]["average_score"] <= 0.5
    ]

    return question_ids


if __name__ == "__main__":
    examples = find_AOK_training_examples_less_than_0_5_pass_rate()
    print(len(examples))
