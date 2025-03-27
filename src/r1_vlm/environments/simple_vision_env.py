import json
from typing import Any, Dict, List, Sequence, Union

from qwen_vl_utils import process_vision_info
from verifiers import SimpleEnv
from vllm import LLM, SamplingParams  # type: ignore


class SimpleVisionEnv(SimpleEnv):
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

        # get completions
        completions = vlm.generate(
            vlm_inputs, sampling_params=custom_sp, use_tqdm=False
        )  # type: ignore

        for i, completion in enumerate(completions):
            states[i]["messages"].append(
                {"role": "assistant", "content": completion.outputs[0].text}
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
            for message in messages:
                cleaned_message = message.copy()
                if "content" in cleaned_message:
                    cleaned_content = []
                    for item in cleaned_message["content"]:
                        cleaned_item = item.copy()
                        if "image" in cleaned_item and cleaned_item["image"] is not None:
                            cleaned_item["image"] = "<PIL.Image object>"
                        cleaned_content.append(cleaned_item)
                    cleaned_message["content"] = cleaned_content
                cleaned.append(cleaned_message)
            return cleaned

        self.logger.info(
            "Prompt 0:\n"
            + json.dumps(clean_messages_for_logging(states[0]["messages"][:-1]), indent=4)
            + "\n\nCompletion 0:\n"
            + json.dumps(states[0]["messages"][-1], indent=4)
        )

        completion_ids = [states[i]["completion_ids"] for i in range(len(states))]
        completion_messages = [states[i]["messages"][-1:] for i in range(len(states))]

        return {
            "ids": completion_ids,
            "messages": completion_messages
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
