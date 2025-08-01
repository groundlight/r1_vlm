import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# load the results file
file_path = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/tool_use_aok_text_vqa_env/eval_on_aok_train_results_8shot.jsonl"

# load the results
with open(file_path, "r") as f:
    results = [json.loads(line) for line in f]
print(f"Loaded {len(results)} results")

# group the results by question_id
results_by_question_id = {}
for result in results:
    if result["question_id"] not in results_by_question_id:
        results_by_question_id[result["question_id"]] = {
            "average_score": 0,
            "results": [],
        }
    results_by_question_id[result["question_id"]]["results"].append(result)

# impute the average score for each question_id
for question_id in results_by_question_id:
    average_score = sum(
        [result["score"] for result in results_by_question_id[question_id]["results"]]
    ) / len(results_by_question_id[question_id]["results"])
    results_by_question_id[question_id]["average_score"] = average_score

# Prepare data for histogram
data = [result["average_score"] for result in results_by_question_id.values()]
total_count = len(data)

# plot as a histogram showing proportions
plt.hist(
    data,
    bins=10,
    weights=[1 / total_count] * total_count,
)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

plt.xlabel("Average Score")
plt.ylabel("Proportion of Questions")
plt.title("8-shot AOK Score Histogram")

plt.savefig("8shot_aok_score_histogram.png")

# count how many questions have an average vqa score <=N
ns = [0.5, 0.4, 0.3, 0.2, 0.1]
for n in ns:
    count = sum(
        1 for result in results_by_question_id.values() if result["average_score"] <= n
    )
    print(f"Number of questions with average score <={n}: {count}")
