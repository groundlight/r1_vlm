import re
from typing import Any, Callable

from datasets import Dataset
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.datasets.text_vqa.text_vqa_tool_use_r1 import (
    create_r1_text_vqa_tool_use_dataset,
)
from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
from r1_vlm.environments.tool_vision_env import ToolArgParser, ToolVisionEnv
from r1_vlm.tools.tool_prompts import SINGLE_TOOL_PROMPT_TEMPLATE
from r1_vlm.tools.zoom import parse_zoom_args, zoom

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

    # if there is only one correct answer, we're done
    if len(set(correct_answers)) == 1:
        return model_answer, correct_answers

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


class TextVQAToolEnv(ToolVisionEnv):
    def __init__(
        self,
        processing_class: AutoProcessor,
        dataset_name: str = None,
        tools_with_parsers: list[tuple[Callable, ToolArgParser]] = [
            (zoom, parse_zoom_args),
        ],
        max_steps: int = 3,
        tool_prompt_template: str = SINGLE_TOOL_PROMPT_TEMPLATE,
    ):
        super().__init__(
            processing_class=processing_class,
            tools_with_parsers=tools_with_parsers,
            max_steps=max_steps,
            tool_prompt_template=tool_prompt_template,
        )

        self.dataset_name = dataset_name
        self.parser = XMLParser(fields=["think", "answer", "tool"])
        self._fields = [
            ("think", ["think"]),
            ("answer", ["answer"]),
            ("tool", ["tool"]),
        ]

    def parse(self, text: str, strip: bool = True):
        return self.parser.parse(text, strip=strip)

    def get_dataset(
        self,
        splits: list[str] = None,
        max_examples_per_split: int | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        dataset = create_r1_text_vqa_tool_use_dataset(
            splits_to_process=splits, max_examples_per_split=max_examples_per_split
        )

        output_datasets = {}
        for split in splits:
            dataset_split = dataset[split]

            # inject system prompt
            dataset_split = self.inject_system_prompt(dataset_split)

            # preprocess the dataset
            dataset_split = preprocess_r1_dataset(dataset_split)

            output_datasets[split] = dataset_split

        if "train" in splits:
            output_datasets["train"].shuffle()

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
            elif reward_function.__name__ == "tool_execution_reward_func":
                # No reward for tool execution
                schedule = 0.0
                reward_weights.append(schedule)

            elif reward_function.__name__ == "correct_answer_reward_func":
                # consistent high reward for getting the answer right
                schedule = 1.0
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
            Expected format: <think>...</think> followed by either <tool>...</tool> or <answer>...</answer>"""

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
                    has_tool = (
                        hasattr(parsed, "tool") and getattr(parsed, "tool") is not None
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
                    if has_tool and not (
                        hasattr(parsed_no_strip, "tool")
                        and getattr(parsed_no_strip, "tool") is not None
                    ):
                        has_correct_spacing = False
                    if has_answer and not (
                        hasattr(parsed_no_strip, "answer")
                        and getattr(parsed_no_strip, "answer") is not None
                    ):
                        has_correct_spacing = False

                    # Check structural validity: think + (tool XOR answer)
                    is_valid_structure = has_think and (
                        (has_tool and not has_answer) or (not has_tool and has_answer)
                    )

                    # Check start and end tags based on expected structure
                    text_stripped = text_content.strip()
                    starts_with_think = text_stripped.startswith("<think>")
                    ends_with_tool = text_stripped.endswith("</tool>")
                    ends_with_answer = text_stripped.endswith("</answer>")

                    # Valid end requires ending with the tag that is present (tool or answer)
                    valid_end = (has_tool and ends_with_tool) or (
                        has_answer and ends_with_answer
                    )

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
                    if valid_end:  # Should end with tool/answer if structure is valid
                        format_score += 0.2

                    format_scores.append(format_score)
                if not format_scores:
                    return 0.0
                # Return the average score over all assistant messages in the trajectory
                return sum(format_scores) / len(format_scores)

            preprocessed_data = TextVQAToolEnv.preprocess_for_reward(
                prompts=prompts,
                completions=completions,
                completions_messages=completions_messages,
            )

            merged_completion_conversations = preprocessed_data[
                "merged_completion_conversations"
            ]

            return [check_format(m) for m in merged_completion_conversations]

        def check_execution(conversation):
            """
            Returns the ratio of successful tool executions to total attempts.
            """
            tool_attempts = 0
            successful_executions = 0

            for i, message in enumerate(conversation):
                if message["role"] == "assistant":
                    parsed = self.parser.parse(message["content"][0]["text"])
                    if hasattr(parsed, "tool") and parsed.tool is not None:
                        tool_attempts += 1
                        if (
                            i + 1 < len(conversation)
                            and conversation[i + 1]["role"] == "user"
                        ):
                            response = conversation[i + 1]["content"][0]["text"]
                            if "Error:" not in response:
                                successful_executions += 1

            return 0.0 if tool_attempts == 0 else successful_executions / tool_attempts

        def tool_execution_reward_func(
            prompts, completions, completions_messages, **kwargs
        ) -> list[float]:
            """
            Reward function that checks if tools were executed successfully.
            Returns a reward based on the ratio of successful tool executions to total attempts.
            """
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
                prompts_messages=prompts, completions_messages=completions_messages
            )

            rewards: list[float] = [
                check_execution(conv) for conv in merged_completion_conversations
            ]

            return rewards

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
            Provides a reward if the model's answer is correct. Gates on the model either not using tools or using tools correctly.
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

        return [
            format_reward_func,
            tool_execution_reward_func,
            correct_answer_reward_func,
        ]

    def count_tool_attempts(self, conversation):
        """
        Returns the number of tool attempts in the conversation.
        """
        tool_attempts = 0
        successful_executions = 0

        for i, message in enumerate(conversation):
            if message["role"] == "assistant":
                parsed = self.parser.parse(message["content"][0]["text"])
                if hasattr(parsed, "tool") and parsed.tool is not None:
                    tool_attempts += 1
                    if (
                        i + 1 < len(conversation)
                        and conversation[i + 1]["role"] == "user"
                    ):
                        response = conversation[i + 1]["content"][0]["text"]
                        if "Error:" not in response:
                            successful_executions += 1

        return tool_attempts

    def log_metrics(self, conversations, completions_text, completion_messages):
        # 1. compute how many completions attempt to use any tool
        # 2. for each tool, compute how many completions attempt to use it

        completions_with_tool_use = 0
        completions_with_zoom_use = 0

        for completion in completions_text:
            tool_use_regex = r"<tool>(.*?)</tool>"
            zoom_use_string = "name: zoom"
            tool_matches = re.findall(tool_use_regex, completion, re.DOTALL)
            if tool_matches:
                completions_with_tool_use += 1
                for tool_content in tool_matches:
                    if zoom_use_string in tool_content:
                        completions_with_zoom_use += 1

        print(
            f"There are {len(completions_text)} completions, {completions_with_tool_use} of which attempt to use a tool, {completions_with_zoom_use} of which attempt to use zoom"
        )

        num_completions = len(completions_text)
        tool_use_proportion = completions_with_tool_use / num_completions
        zoom_use_proportion = completions_with_zoom_use / num_completions

        # I want to measure if any completion has more than one tool use
        tool_uses_per_completion = [
            self.count_tool_attempts(messages) for messages in completion_messages
        ]

        any_completion_with_more_than_one_tool_use = any(
            tool_use > 1 for tool_use in tool_uses_per_completion
        )

        return {
            "tool_use_proportion": tool_use_proportion,
            "zoom_use_proportion": zoom_use_proportion,
            "any_completion_with_more_than_one_tool_use": any_completion_with_more_than_one_tool_use,
        }


if __name__ == "__main__":
    env = TextVQAToolEnv(processing_class=None)
    train_dataset, val_dataset, test_dataset = env.get_dataset()
    import ipdb

    ipdb.set_trace()
    print("hi")
