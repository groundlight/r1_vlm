import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# load the results file
file_path = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/tool_use_text_vqa_env/eval_on_train_results_8shot.jsonl"

# load the results
with open(file_path, "r") as f:
    results = [json.loads(line) for line in f]
print(f"Loaded {len(results)} results")

# group the results by question_id
results_by_question_id = {}
for result in results:
    if result["question_id"] not in results_by_question_id:
        results_by_question_id[result["question_id"]] = {
            "average_vqa_score": 0,
            "results": [],
        }
    results_by_question_id[result["question_id"]]["results"].append(result)

# impute the average vqa score for each question_id
for question_id in results_by_question_id:
    average_vqa_score = sum(
        [
            result["vqa_score"]
            for result in results_by_question_id[question_id]["results"]
        ]
    ) / len(results_by_question_id[question_id]["results"])
    results_by_question_id[question_id]["average_vqa_score"] = average_vqa_score

# Prepare data for histogram
data = [result["average_vqa_score"] for result in results_by_question_id.values()]
total_count = len(data)

# plot as a histogram showing proportions
plt.hist(
    data,
    bins=10,
    weights=[1 / total_count] * total_count,
)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

plt.xlabel("Average VQA Score")
plt.ylabel("Proportion of Questions")
plt.title("8-shot VQA Score Histogram")

plt.savefig("8shot_vqa_score_histogram.png")

# count how many questions have an average vqa score <=N
ns = [0.5, 0.4, 0.3, 0.2, 0.1]
for n in ns:
    count = sum(
        1
        for result in results_by_question_id.values()
        if result["average_vqa_score"] <= n
    )
    print(f"Number of questions with average vqa score <={n}: {count}")

# compute overall vqa score
overall_vqa_score = sum(
    [result["average_vqa_score"] for result in results_by_question_id.values()]
) / len(results_by_question_id)
print(f"Overall VQA score: {overall_vqa_score}")
