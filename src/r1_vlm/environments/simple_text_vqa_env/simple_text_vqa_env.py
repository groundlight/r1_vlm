import json
import re
from typing import Any

import Levenshtein
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.datasets.text_vqa.text_vqa_r1 import (
    create_r1_text_vqa_dataset,
)
from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
from r1_vlm.environments.simple_vision_env import SimpleVisionEnv
from r1_vlm.environments.tool_use_text_vqa_env.find_examples_for_training import (
    find_examples_for_training,
)

# Implementing the exact preprocessing steps that are used in text vqa evaluation
# https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L11
contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
manualMap = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def normalize_answer(model_answer, correct_answers):
    """
    implements the normalization logic from the text vqa eval that normalizes the answer and correct answers
    """
    # always remove new lines from the answer and correct answers
    model_answer = model_answer.replace("\n", " ")
    correct_answers = [answer.replace("\n", " ") for answer in correct_answers]

    # always remote "\t" from the answer and correct answers
    model_answer = model_answer.replace("\t", " ")
    correct_answers = [answer.replace("\t", " ") for answer in correct_answers]

    # always strip answer and correct answers
    model_answer = model_answer.strip()
    correct_answers = [answer.strip() for answer in correct_answers]

    # # if there is only one correct answer, we're done
    # if len(set(correct_answers)) == 1:
    #     return model_answer, correct_answers

    # otherwise we get more normalization, first process punctuation
    correct_answers = [process_punctuation(answer) for answer in correct_answers]
    model_answer = process_punctuation(model_answer)

    # then process digit articles
    correct_answers = [process_digit_articles(answer) for answer in correct_answers]
    model_answer = process_digit_articles(model_answer)

    return model_answer, correct_answers


# "process" functions are copy pasted from the text vqa eval directly
def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (
            re.search(commaStrip, inText) is not None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def process_digit_articles(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


class SimpleTextVQAEnv(SimpleVisionEnv):
    def __init__(
        self,
        dataset_name: str = None,
    ):
        self.dataset_name = dataset_name
        self.parser = XMLParser(fields=["think", "answer"])
        self._fields = [
            ("think", ["think"]),
            ("answer", ["answer"]),
        ]

    def parse(self, text: str, strip: bool = True):
        return self.parser.parse(text, strip=strip)

    def get_dataset(
        self,
        splits: list[str] = None,
        max_examples_per_split: int | None = None,
        max_size: int = 1024,
        skip_index: int | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        # only use examples where zeroshot VQA score is 0.0
        train_examples_to_include = find_examples_for_training()

        dataset = create_r1_text_vqa_dataset(
            splits_to_process=splits,
            max_examples_per_split=max_examples_per_split,
            max_size=max_size,
            train_examples_to_include=train_examples_to_include,
        )

        output_datasets = {}
        for split in splits:
            dataset_split = dataset[split]

            # preprocess the dataset
            dataset_split = preprocess_r1_dataset(dataset_split)

            output_datasets[split] = dataset_split

        if "train" in splits:
            output_datasets["train"].shuffle()

            if skip_index is not None:
                output_datasets["train"] = output_datasets["train"].select(
                    range(skip_index, len(output_datasets["train"]))
                )

        return output_datasets

    @staticmethod
    def preprocess_for_reward(*, prompts, completions, completions_messages):
        """
        Preprocessing for rewards

        Returns: dict with keys
        merged_completion_conversations: list[dict] - a list of the completion conversations (after merging)
        """
        merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
            prompts_messages=prompts, completions_messages=completions_messages
        )

        return {
            "merged_completion_conversations": merged_completion_conversations,
        }

    @staticmethod
    def get_assistant_messages(conversation: list[dict[str, Any]]) -> list[str]:
        """
        Returns the assistant messages from the completion messages as a list of strings.
        """
        assistant_messages = [
            message["content"][0]["text"]
            for message in conversation
            if message["role"] == "assistant"
        ]
        return assistant_messages

    def get_reward_weights(self) -> list[float]:
        reward_functions = self.get_rubric()
        reward_weights = []
        for reward_function in reward_functions:
            if reward_function.__name__ == "format_reward_func":
                schedule = 1.0
                reward_weights.append(schedule)
            elif reward_function.__name__ == "correct_answer_reward_func":
                # correctness reward is split between soft and hard correctness
                schedule = 0.5
                reward_weights.append(schedule)
            elif reward_function.__name__ == "soft_correctness_reward_func":
                schedule = 0.5
                reward_weights.append(schedule)
            else:
                raise ValueError(
                    f"Unknown reward function: {reward_function.__name__} encountered in get_reward_weights"
                )

        assert len(reward_weights) == len(reward_functions), (
            f"reward_weights and reward_functions should be the same length, but got {len(reward_weights)} and {len(reward_functions)}"
        )
        return reward_weights

    def get_rubric(self) -> list[RewardFunc]:
        def format_reward_func(prompts, completions, completions_messages, **kwargs):
            """Soft reward function that checks if each step follows the expected format.
            Expected format: <think>...</think> followed by <answer>...</answer>"""

            def check_format(trajectory):
                model_messages = [
                    msg for msg in trajectory if msg["role"] == "assistant"
                ]
                if not model_messages:
                    return 0.0
                format_scores = []
                for msg in model_messages:
                    content = msg["content"]
                    if isinstance(content, list):
                        text_content = "".join(part.get("text", "") for part in content)
                    else:
                        text_content = content
                    parsed = self.parse(text_content)
                    parsed_no_strip = self.parse(text_content, strip=False)

                    # Check specific field presence
                    has_think = (
                        hasattr(parsed, "think")
                        and getattr(parsed, "think") is not None
                    )
                    has_answer = (
                        hasattr(parsed, "answer")
                        and getattr(parsed, "answer") is not None
                    )

                    # Check correct spacing (no extra whitespace within tags)
                    # This requires checking non-stripped parsing results for *present* fields
                    has_correct_spacing = True
                    if has_think and not (
                        hasattr(parsed_no_strip, "think")
                        and getattr(parsed_no_strip, "think") is not None
                    ):
                        has_correct_spacing = False
                    if has_answer and not (
                        hasattr(parsed_no_strip, "answer")
                        and getattr(parsed_no_strip, "answer") is not None
                    ):
                        has_correct_spacing = False

                    # Check structural validity: think + (tool XOR answer)
                    is_valid_structure = has_think and has_answer

                    # Check start and end tags based on expected structure
                    text_stripped = text_content.strip()
                    starts_with_think = text_stripped.startswith("<think>")
                    ends_with_answer = text_stripped.endswith("</answer>")

                    # Valid end requires ending with the tag that is present (tool or answer)
                    valid_end = has_answer and ends_with_answer

                    # Calculate score
                    format_score = 0.0
                    if is_valid_structure:
                        format_score += 0.4  # Core structure reward
                    if has_correct_spacing:
                        format_score += 0.2
                    if (
                        starts_with_think
                    ):  # Should always start with think if structure is valid
                        format_score += 0.2
                    if valid_end:  # Should end with answer if structure is valid
                        format_score += 0.2

                    format_scores.append(format_score)
                if not format_scores:
                    return 0.0
                # Return the average score over all assistant messages in the trajectory
                return sum(format_scores) / len(format_scores)

            preprocessed_data = SimpleTextVQAEnv.preprocess_for_reward(
                prompts=prompts,
                completions=completions,
                completions_messages=completions_messages,
            )

            merged_completion_conversations = preprocessed_data[
                "merged_completion_conversations"
            ]

            return [check_format(m) for m in merged_completion_conversations]

        def check_correctness(conversation, correct_answers) -> bool:
            text = conversation[-1]["content"][0]["text"]

            parsed = self.parser.parse(text)
            if hasattr(parsed, "answer") and parsed.answer is not None:
                answer = parsed.answer
            else:
                return 0.0

            print(
                f"before normalization, answer: {answer}, correct_answers: {correct_answers}"
            )

            answer, correct_answers = normalize_answer(answer, correct_answers)

            print(
                f"after normalization, answer: {answer}, correct_answers: {correct_answers}"
            )

            # check how many correct answers match the given answer
            num_matches = sum(
                1 for correct_answer in correct_answers if correct_answer == answer
            )

            score = min(1, num_matches / 3)
            return score

        def correct_answer_reward_func(
            prompts, completions, completions_messages, **kwargs
        ) -> list[float]:
            """
            Provides a reward if the model's answer is correct.
            """
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
                prompts_messages=prompts, completions_messages=completions_messages
            )

            # a list of lists of 10 responses from labelers. They should all be the same. Verify this, then take the first one.
            correct_answers_lists = kwargs["answers"]
            for correct_answers_list in correct_answers_lists:
                if not correct_answers_list == correct_answers_lists[0]:
                    raise ValueError(
                        f"All correct answers lists should be the same, but got {correct_answers_list}"
                    )
            correct_answers = correct_answers_lists[0]

            correctness_rewards = []
            for conv in merged_completion_conversations:
                # the correct answers remain constant across the dataset
                correctness_rewards.append(check_correctness(conv, correct_answers))

            # verify all rewards between 0 and 1
            assert all(0 <= reward <= 1 for reward in correctness_rewards), (
                f"All rewards should be between 0 and 1, but got {correctness_rewards}"
            )

            return correctness_rewards

        def get_edit_distance_score(conversation, correct_answers):
            """
            Returns a "soft" vqa score based on the edit distance between the model's answer and the correct answers.
            """
            text = conversation[-1]["content"][0]["text"]

            parsed = self.parser.parse(text)
            if hasattr(parsed, "answer") and parsed.answer is not None:
                answer = parsed.answer
            else:
                return 0.0

            # normalize the str data so we are consistent with vqa score
            answer, correct_answers = normalize_answer(answer, correct_answers)

            # compute the normalized edit distance between the answer and each of the correct answers
            normalized_edit_distances = []

            for correct_answer in correct_answers:
                try:
                    edit_distance = Levenshtein.distance(answer, correct_answer)
                    normalized_edit_distance = edit_distance / max(
                        len(answer), len(correct_answer)
                    )
                    normalized_edit_distances.append(normalized_edit_distance)
                except Exception:
                    # if both the answer and correct_answer have length 0, the norm fails. In this case, we are conservative and give a score of 1.0, implying the model answer is very wrong
                    normalized_edit_distances.append(1.0)

            # lower edit distance = higher score
            normalized_scores = [
                1 - normalized_edit_distance
                for normalized_edit_distance in normalized_edit_distances
            ]

            # take the top 3 scores (inspired by VQA score) and average them, clip at 1
            top_3_scores = sorted(normalized_scores, reverse=True)[:3]
            score = min(1, sum(top_3_scores) / 3)

            print(
                f"answer: {answer}\ncorrect_answers: {correct_answers}\nnormalized_edit_distances: {normalized_edit_distances}\nnormalized_scores: {normalized_scores}\ntop_3_scores: {top_3_scores}\nscore: {score}"
            )

            return score

        def soft_correctness_reward_func(
            prompts, completions, completions_messages, **kwargs
        ) -> list[float]:
            """
            Reward function that checks if the model's answer is correct. Uses a soft normalized edit distance to measure correctness.
            """
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
                prompts_messages=prompts, completions_messages=completions_messages
            )

            # a list of lists of 10 responses from labelers. They should all be the same. Verify this, then take the first one.
            correct_answers_lists = kwargs["answers"]
            for correct_answers_list in correct_answers_lists:
                if not correct_answers_list == correct_answers_lists[0]:
                    raise ValueError(
                        f"All correct answers lists should be the same, but got {correct_answers_list}"
                    )
            correct_answers = correct_answers_lists[0]

            soft_correctness_rewards = []
            for conv in merged_completion_conversations:
                # the correct answers remain constant across the dataset
                soft_correctness_rewards.append(
                    get_edit_distance_score(conv, correct_answers)
                )

            # verify all rewards between 0 and 1
            assert all(0 <= reward <= 1 for reward in soft_correctness_rewards), (
                f"All rewards should be between 0 and 1, but got {soft_correctness_rewards}"
            )

            return soft_correctness_rewards

        return [
            format_reward_func,
            correct_answer_reward_func,
            soft_correctness_reward_func,
        ]

    def log_metrics(
        self,
        conversations,
        completions_text,
        completion_messages,
        advantages,
        global_step,
    ):
        # for each completion we want to log:
        # 1. the advantage
        # 2. classifiy the completion as:
        # - no tool use
        # 3. the global step

        for advantage, messages in zip(advantages, completion_messages):
            classification = "no_tool_use"
            # save to a jsonl file
            with open(
                "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/simple_text_vqa_env/simple_text_vqa_metrics.jsonl",
                "a",
            ) as f:
                data_point = {
                    "advantage": advantage,
                    "classification": classification,
                    "global_step": global_step,
                }
                f.write(json.dumps(data_point) + "\n")

        return {}


if __name__ == "__main__":
    env = SimpleTextVQAEnv(processing_class=None)
    train_dataset, val_dataset, test_dataset = env.get_dataset()
    import ipdb

    ipdb.set_trace()
    print("hi")
