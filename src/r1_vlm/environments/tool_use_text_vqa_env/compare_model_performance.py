import json

from tqdm import tqdm

from r1_vlm.environments.tool_use_text_vqa_env.tool_use_text_vqa_env import (
    normalize_answer,
)


def add_vqa_scores(results: list[dict]):
    for result in results:
        if "vqa_score" in result:
            continue

        model_answer = result["model_answer"]
        correct_answers = result["answers"]
        model_answer, correct_answers = normalize_answer(model_answer, correct_answers)

        num_matches = sum(
            1 for correct_answer in correct_answers if correct_answer == model_answer
        )

        score = min(1, num_matches / 3)

        result["vqa_score"] = score

    return results


if __name__ == "__main__":
    # zoom model
    trained_model_eval = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/tool_use_text_vqa_env/eval_zoom_vqa_less_than_0_5_with_small_tool_reward_step650_validation_results_single_shot.jsonl"

    # baseline model
    # trained_model_eval = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/simple_text_vqa_env/eval_baseline_step800_validation_results.jsonl"

    # base model
    base_model_eval = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/tool_use_text_vqa_env/base_eval_on_validation_results.jsonl"

    base_model_results = []
    with open(base_model_eval, "r") as f:
        for line in f:
            base_model_results.append(json.loads(line))

    trained_model_results = []
    with open(trained_model_eval, "r") as f:
        for line in f:
            trained_model_results.append(json.loads(line))

    trained_model_results = add_vqa_scores(trained_model_results)

    print(f"length of base model results: {len(base_model_results)}")
    print(f"length of trained model results: {len(trained_model_results)}")

    # we'll only evaluate the results that are present in both the trained and base model results (this is a safety check as we are still evaluating)
    # once eval is done, we might just check that all the question ids are present in the results of both files
    trained_question_ids = [result["question_id"] for result in trained_model_results]
    base_question_ids = [result["question_id"] for result in base_model_results]
    question_ids_to_evaluate = set(trained_question_ids) & set(base_question_ids)
    print(f"length of question ids to evaluate: {len(question_ids_to_evaluate)}")

    # for each question, collect the results from both the trained and base model results
    all_results = []
    for question_id in tqdm(
        question_ids_to_evaluate, desc="Collating evaluation results"
    ):
        # get the result for the base model and the trained model
        base_result = next(
            (
                result
                for result in base_model_results
                if result["question_id"] == question_id
            ),
            None,
        )
        assert base_result is not None, (
            f"Base model result not found for question id: {question_id}"
        )

        # get the result for the trained model
        trained_result = next(
            (
                result
                for result in trained_model_results
                if result["question_id"] == question_id
            ),
            None,
        )
        assert trained_result is not None, (
            f"Trained model result not found for question id: {question_id}"
        )

        # collate them into a single dictionary
        result = {
            "question_id": question_id,
            "base_model_result": base_result["vqa_score"],
            "trained_model_result": trained_result["vqa_score"],
        }
        all_results.append(result)

    # first consider the overall evaluation scores for both models
    overall_base_score = sum(
        result["base_model_result"] for result in all_results
    ) / len(all_results)
    overall_trained_score = sum(
        result["trained_model_result"] for result in all_results
    ) / len(all_results)

    print(f"Overall base model score: {overall_base_score}")
    print(f"Overall trained model score: {overall_trained_score}")

    # now only consider the elements where the base model scores 0.0, we want to track both the score and number of elements
    total_base_examples_with_zero_score = 0
    total_trained_model_score_on_base_examples_with_zero_score = 0
    for result in all_results:
        if result["base_model_result"] <= 0.0:
            total_base_examples_with_zero_score += 1
            total_trained_model_score_on_base_examples_with_zero_score += result[
                "trained_model_result"
            ]

    print(
        f"Average trained model score on base examples with zero score: {total_trained_model_score_on_base_examples_with_zero_score / total_base_examples_with_zero_score} on {total_base_examples_with_zero_score} examples"
    )

    # plot the trained model's score on the zero score examples over the course of evaluation
    datapoints = []
    running_score = None
    total_count = 0
    for result in all_results:
        if result["base_model_result"] > 0.0:
            continue

        total_count += 1

        score = result["trained_model_result"]

        if running_score is None:
            running_score = score
        else:
            running_score += score

        datapoints.append(running_score / total_count)

    from imgcat import imgcat
    from matplotlib import pyplot as plt

    xs = list(range(total_count))
    plt.plot(xs, datapoints)
    plt.savefig("/tmp/plot.png")
    with open("/tmp/plot.png", "rb") as img_file:
        imgcat(img_file.read())
