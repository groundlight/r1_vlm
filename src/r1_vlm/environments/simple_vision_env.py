import json
import random
from typing import Any, Dict, List, Sequence, Union

import imgcat
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams  # type: ignore

from r1_vlm.budget_forcing.budget_forcing import (
    generate_completions_with_budget_forcing,
)
from verifiers import SimpleEnv


class SimpleVisionEnv(SimpleEnv):
    def __init__(
        self,
        use_budget_forcing: bool = False,
        max_thinking_tokens: int = 1024,
        num_ignore: int | list[int] = 1,
        ignore_str: str | list[str] = "Wait",
        **kwargs: Any,
    ):
        """
        Initialize the SimpleVisionEnv.

        use_budget_forcing: bool = False, Whether to use budget forcing. If true, we will budget force the model to think more with the following parameters:

        max_thinking_tokens: int = 1024, The maximum number of tokens the model can think for.

        num_ignore: int| list[int] = 1, The number of times we're willing to ignore the model trying to end thinking. If a list is provided, we will draw randomly from the list each time we generate.

        ignore_str: str| list[str] = "Wait", The string we manually add when the model tries to end thinking to promote the model to think more. If a list is provided, we will draw randomly from the list each time we need an ignore string during generation.
        """
        super().__init__(**kwargs)
        self.use_budget_forcing = use_budget_forcing
        self.max_thinking_tokens = max_thinking_tokens

        if isinstance(ignore_str, str):
            self.ignore_str = [ignore_str]
        elif isinstance(ignore_str, list):
            self.ignore_str = ignore_str
        else:
            raise ValueError(
                f"ignore_str must be a str or a list of strs, got {type(ignore_str)}"
            )

        if isinstance(num_ignore, int):
            self.num_ignore = [num_ignore]
        elif isinstance(num_ignore, list):
            self.num_ignore = num_ignore
        else:
            raise ValueError(
                f"num_ignore must be an int or a list of ints, got {type(num_ignore)}"
            )

    def generate(
        self,
        conversations,
        vlm_inputs,  # TODO: Add type
        vlm: LLM,
        sampling_params: SamplingParams,
        **kwargs: Any,
    ) -> Union[List[Sequence[int]], List[str], List[List[Dict[str, Any]]]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        states = [
            {
                # a list of conversations
                "messages": conversation,
                "prompt_ids": [],
                "completion_ids": [],
            }
            for conversation in conversations
        ]

        # generate completions either through budget forcing or just using vllm directly
        if self.use_budget_forcing:
            # choose a number from self.num_ignore for this generation
            num_ignore = random.choice(self.num_ignore)

            completions = generate_completions_with_budget_forcing(
                vllm_inputs=vlm_inputs,
                vlm=vlm,
                processor=self.processing_class,
                max_thinking_tokens=self.max_thinking_tokens,
                num_ignore=num_ignore,
                ignore_str=self.ignore_str,
            )
        else:
            completions = vlm.generate(
                vlm_inputs, sampling_params=custom_sp, use_tqdm=False
            )  # type: ignore
            
            stop_reasons = [c.outputs[0].stop_reason for c in completions]
            print(f"Stop reasons: {stop_reasons}")

        for i, completion in enumerate(completions):
            states[i]["messages"].append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": completion.outputs[0].text}],
                }
            )
            states[i]["prompt_ids"] = list(completion.prompt_token_ids)
            states[i]["completion_ids"] = list(completion.outputs[0].token_ids)

        self.logger.debug(
            f"Prompt 0 IDs: {states[0]['prompt_ids']} \nlen: {len(states[0]['prompt_ids'])}"
        )
        self.logger.debug(
            f"Completion 0 IDs: {states[0]['completion_ids']} \nlen: {len(states[0]['completion_ids'])}"
        )

        def clean_messages_for_logging(messages):
            cleaned = []
            images = []
            for message in messages:
                cleaned_message = message.copy()
                if "content" in cleaned_message:
                    cleaned_content = []
                    for item in cleaned_message["content"]:
                        cleaned_item = item.copy()
                        if (
                            "image" in cleaned_item
                            and cleaned_item["image"] is not None
                        ):
                            images.append(cleaned_item["image"])
                            cleaned_item["image"] = "<PIL.Image object>"
                        cleaned_content.append(cleaned_item)
                    cleaned_message["content"] = cleaned_content
                cleaned.append(cleaned_message)
            return cleaned, images

        cleaned_messages, images = clean_messages_for_logging(states[0]["messages"])

        self.logger.info(
            "Prompt 0:\n"
            + json.dumps(cleaned_messages, indent=4)
            + "\n\nCompletion 0:\n"
            + json.dumps(states[0]["messages"][-1], indent=4)
        )
        for image in images:
            imgcat.imgcat(image)

        completion_ids = [states[i]["completion_ids"] for i in range(len(states))]
        completion_messages = [states[i]["messages"][-1:] for i in range(len(states))]
        # Create masks of 1's matching the length of each completion
        completion_masks = [[1] * len(ids) for ids in completion_ids]

        return {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_masks,
        }

    def prepare_data(self, *, inputs, processing_class):
        """
        prepares the data to be used for forward pass with VLLM and logprobs calculations with hf
        """
        conversations, texts, batch, vllm_inputs = prepare_inputs_for_env(
            inputs=inputs, processing_class=processing_class
        )

        return conversations, texts, batch, vllm_inputs


def prepare_inputs_for_env(*, inputs, processing_class):
    """
    Prepares inputs for an env's .generate method.

    inputs: a list of inputs, in this case a list of examples from our dataset
    processing_class: the processing class to use to process the inputs. This is a VLM processor object from the transformers library.
    """

    # WARNING: This is used in SimpleVisionEnv AND MultistepVisionEnv. If you are making changes here, be aware that it will affect both.

    conversations = [ex["messages"] for ex in inputs]

    # Clean up None values from the messages
    for conv in conversations:
        for message in conv:
            content = message["content"]
            message["content"] = [
                {k: v for k, v in item.items() if v is not None} for item in content
            ]

    # apply the chat template to the messages and add image tokens
    texts = processing_class.apply_chat_template(
        conversations, continue_final_message=True, tokenize=False
    )

    vllm_inputs = []
    for conversation, text in zip(conversations, texts):
        vllm_image_inputs, _ = process_vision_info(conversation)
        mm_data = {"image": vllm_image_inputs}
        vllm_input = {"prompt": text, "multi_modal_data": mm_data}
        vllm_inputs.append(vllm_input)

    batch_image_inputs = []
    for conv in conversations:
        image_input, _ = process_vision_info(conv)
        batch_image_inputs.append(image_input)

    batch = processing_class(
        text=texts,
        images=batch_image_inputs,
        padding=True,
        return_tensors="pt",
    )

    return conversations, texts, batch, vllm_inputs
