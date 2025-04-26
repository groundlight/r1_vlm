import inspect
import traceback
from typing import Any, Callable, Dict, List

from datasets import Dataset
from PIL.Image import Image
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
from r1_vlm.tools.tool_prompts import DEFAULT_TOOL_PROMPT_TEMPLATE


def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()

    # Extract examples if present
    examples = []
    for part in doc_parts:
        if part.strip().startswith("Examples:"):
            part_lines = part.split("\n")
            examples_line_index = -1
            for i, line in enumerate(part_lines):
                if line.strip().startswith("Examples:"):
                    examples_line_index = i
                    break

            if examples_line_index != -1:
                lines_in_part = [
                    line.strip()
                    for line in part_lines[examples_line_index + 1 :]
                    if line.strip()
                ]
                examples.extend(lines_in_part)

    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        # Skip kwargs parameter
        if name == "kwargs":
            continue

        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name) + 1 :].strip()

        args[name] = {
            "type": str(
                param.annotation.__name__
                if param.annotation != inspect.Parameter.empty
                else "any"
            ),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default

    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": str(
            sig.return_annotation.__name__
            if sig.return_annotation != inspect.Parameter.empty
            else "any"
        ),
        "examples": examples,
    }


def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]

        desc.append("\nArguments:")
        for arg_name, arg_info in schema["args"].items():
            default = (
                f" (default: {arg_info['default']})" if "default" in arg_info else ""
            )
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")

        if schema["examples"]:
            desc.append("\nExamples:")
            for example in schema["examples"]:
                desc.append(f"  {example}")

        descriptions.append("\n".join(desc))

    return "\n\n".join(descriptions)


# Define the expected structure for the general parser's output
RawToolArgs = Dict[str, str]
# Define the expected structure for the specific parser's output
TypedToolArgs = Dict[str, Any]
# Define the type hint for the specific parser function
ToolArgParser = Callable[[RawToolArgs], TypedToolArgs]


class ToolVisionEnv(MultistepVisionEnv):
    def __init__(
        self,
        # Expect a list of (tool_function, tool_argument_parser) tuples
        tools_with_parsers: List[tuple[Callable, ToolArgParser]],
        processing_class: AutoProcessor,
        sampling_args={
            "stop": ["</tool>", "</answer>"],
            "include_stop_str_in_output": True,
        },
        mask_env_response: bool = True,
        max_steps: int = 10,
        tool_prompt_template: str = DEFAULT_TOOL_PROMPT_TEMPLATE,
        # Allow passing a custom general parser if needed, default to _general_parse_key_value
        general_parser: Callable[[str], RawToolArgs] | None = None,
        **kwargs,
    ):
        super().__init__(
            processing_class=processing_class,
            sampling_args=sampling_args,
            mask_env_response=mask_env_response,
            **kwargs,
        )

        # Store tools and parsers, keyed by tool name
        self.tools: Dict[str, Callable] = {}
        self.tool_parsers: Dict[str, ToolArgParser] = {}
        self.tool_schemas: List[Dict[str, Any]] = []

        for tool_func, parser_func in tools_with_parsers:
            tool_name = tool_func.__name__
            if tool_name in self.tools:
                raise ValueError(f"Duplicate tool name found: {tool_name}")
            self.tools[tool_name] = tool_func
            self.tool_parsers[tool_name] = parser_func
            # Schema inference still uses the tool function's signature/docstring
            self.tool_schemas.append(infer_schema_from_function(tool_func))

        # Format the system prompt with tool descriptions
        tool_descriptions = format_tool_descriptions(self.tool_schemas)
        formatted_prompt = tool_prompt_template.format(
            tool_descriptions=tool_descriptions
        )
        self.formatted_prompt = formatted_prompt

        # Set the general parser (use internal default if none provided)
        self.general_parser = general_parser or self._general_parse_key_value

        # will be used to parse responses from the model. Each response is expected to have a "think" and either a
        # "tool" or "answer" field.
        self.llm_parser = XMLParser(fields=["think", ("tool", "answer", "chars")])

        # will be used to format responses from the environment to return to the model. Tool responses are expected to
        # have a "result" field.
        self.env_parser = XMLParser(fields=["result"])

        self.image_name_parser = XMLParser(fields=["image_name"])

        # the maximum number of assistant messages allowed in the conversation before we end it.
        # can end early if the model provides an answer.
        self.max_steps = max_steps

    def inject_system_prompt(self, dataset: Dataset) -> Dataset:
        """
        Called by inherited class to inject a system prompt containing tool schemas into the given dataset.

        Expects a dataset with a "messages" column. If the first message is a system message, it will be replaced with the formatted prompt.
        Otherwise, this will raise an error. This implementation uses an eager map on the dataset.

        Returns the modified dataset.
        """

        def _inject_prompt(examples):
            messages_batch = examples["messages"]

            for messages in messages_batch:
                if not messages or messages[0]["role"] != "system":
                    raise ValueError("Expected first message to be a system message")

                # Replace the content of the system message with the formatted prompt
                messages[0]["content"] = [
                    {
                        "type": "text",
                        "text": self.formatted_prompt,
                    }
                ]

            return examples

        return dataset.map(_inject_prompt, batched=True)

    def get_rubric(self) -> List[RewardFunc]:
        raise NotImplementedError(
            "ToolVisionEnv requires a rubric for your task. Expected to be implemented by subclass."
        )

    def _conversation_to_image_dict(
        self, conversation: List[Dict[str, Any]]
    ) -> Dict[str, Image]:
        """
        Converts a conversation to a dictionary of image names and PIL images. This dict will be provided as a kwarg to all tools.
        """

        images_list: tuple[str, Image] = []

        # We should look at each element of the conversation. For each element, we should check the content.
        # If we find an image, we should find the image name one message before it in the same content block.

        for element in conversation:
            content = element["content"]
            for index in range(len(content)):
                # look at the current message, is it an image? If it is, we should find the image name one message before it in the same content block.
                message = content[index]
                if message["type"] == "image" and message["image"] is not None:
                    image_name = self.image_name_parser.parse(
                        content[index - 1]["text"]
                    ).image_name
                    images_list.append((image_name, message["image"]))

        # check that all image names are unique
        image_names = set(image_name for image_name, _ in images_list)

        if len(image_names) != len(images_list):
            raise ValueError(
                "All image names must be unique. Found duplicate image names: "
                + str(image_names)
            )

        return dict(images_list)

    def call_tool(
        self,
        tool_text_blob: str,
        messages: List[Dict[str, str]],
        images: Dict[str, Image],
        **kwargs: Any,
    ) -> str | Image | dict:
        """
        Call a tool based on the simple key-value text format.
        1. Use general parser to get raw key-value strings.
        2. Use tool-specific registered parser to validate and type-convert args.
        3. Call the actual tool function.
        """
        tool_name = "unknown"  # Initialize for error messages
        try:
            # Stage 1: General Parsing using the configured general parser
            raw_args = self.general_parser(tool_text_blob)

            tool_name = raw_args.get("name")
            if not tool_name:
                return "Error: Tool call must specify 'name' in key-value format."
            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}' specified. Valid tools are: {', '.join(self.tools.keys())}"

            # Stage 2: Tool-Specific Parsing & Validation (using registered parser)
            if tool_name not in self.tool_parsers:
                # This indicates a configuration error in __init__
                raise ValueError(
                    f"CRITICAL ERROR: Parser not found for registered tool '{tool_name}'. Check __init__ configuration. Existing tool parsers: {self.tool_parsers.keys()}"
                )

            parser_func = self.tool_parsers[tool_name]
            # Pass raw_args (which includes 'name') to the specific parser
            typed_args = parser_func(raw_args)  # This might raise ValueError

            # Stage 3: Call Actual Tool
            tool_func = self.tools[tool_name]
            # Prepare kwargs for the actual tool call (runtime context)
            tool_kwargs = {"messages": messages, "images": images}
            # Call the tool with validated, typed args AND the runtime context
            result = tool_func(**typed_args, **tool_kwargs)

            if isinstance(result, (str, Image, dict)):
                return result
            else:
                return str(result)  # Fallback conversion

        except ValueError as e:  # Catch validation errors from specific parsers
            # Include the specific failing key/value if possible from the error message
            return f"Error: Parsing arguments for tool '{tool_name}': {str(e)}"
        except Exception as e:
            traceback.print_exc()
            # Provide more context in the error returned to the LLM
            return f"Error: During execution of tool '{tool_name}': {str(e)}"

    def env_response(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> list[dict[str, Any]]:
        try:
            last_message = messages[-1]
            # the last message should be an assistant message with text content
            if (
                last_message["role"] != "assistant"
                or last_message["content"][0]["type"] != "text"
            ):
                print(
                    f"Expected last message to be an assistant message and text content: {last_message=}"
                )
                raise ValueError(
                    "Expected last message to be an assistant message and text content"
                )

            # extract the text content from the last message
            message_to_parse = last_message["content"][0]["text"]
            parsed = self.llm_parser.parse(message_to_parse)

            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, "tool") and parsed.tool is not None:
                images = self._conversation_to_image_dict(messages)

                result = self.call_tool(
                    tool_text_blob=parsed.tool, messages=messages, images=images
                )
                if isinstance(result, Image):
                    response = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"<image_name> tool_result_{self._get_step_count(messages) // 2} </image_name>",
                            },
                            {"type": "image", "image": result},
                        ],
                    }
                elif isinstance(result, str) and len(result.strip()) > 0:
                    response = {
                        "role": "user",
                        "content": [
                            {
                                "text": self.env_parser.format(result=result),
                                "type": "text",
                            }
                        ],
                    }
                elif (
                    isinstance(result, dict)
                    and "text_data" in result
                    and "image_data" in result
                ):
                    response = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"<image_name> tool_result_{self._get_step_count(messages) // 2} </image_name>",
                            },
                            {"type": "image", "image": result["image_data"]},
                            {
                                "type": "text",
                                "text": self.env_parser.format(
                                    result=result["text_data"]
                                ),
                            },
                        ],
                    }
                else:
                    response = {
                        "role": "user",
                        "content": [
                            {
                                "text": "Error: Tool execution returned empty output.",
                                "type": "text",
                            }
                        ],
                    }
            elif hasattr(parsed, "answer") and parsed.answer is not None:
                response = {
                    "role": "user",
                    "content": [{"text": parsed.answer, "type": "text"}],
                }
            else:
                response = {
                    "role": "user",
                    "content": [
                        {
                            "text": "Error: No tool call or answer found in your last message.",
                            "type": "text",
                        }
                    ],
                }

        except Exception as e:
            response = {
                "role": "user",
                "content": [
                    {
                        "text": "Error when trying to respond to your last message: "
                        + str(e),
                        "type": "text",
                    }
                ],
            }

        bootstrap = {
            "role": "assistant",
            "content": [{"text": "<think>", "type": "text"}],
        }
        env_response_data = [response, bootstrap]
        return env_response_data

    def _get_step_count(self, messages: List[Dict[str, Any]]) -> int:
        """
        Counts the number of assistant messages in the message history.

        Args:
            messages: the conversation history
        """

        return sum(1 for message in messages if message["role"] == "assistant")

    def is_completed(self, messages: List[Dict[str, Any]], **kwargs: Any) -> bool:
        """
        A completion is finished when either:
        1. We've seen self.max_steps messages from the assistant
        2. The assistant provides an answer.
        """

        step_count = self._get_step_count(messages)

        if step_count >= self.max_steps:
            return True

        parsed = self.llm_parser.parse(messages[-1]["content"][0]["text"])

        # if the last message includes the EOS token, we should end. The model is getting a bad habit of answering without
        # responding at all. We need to penalize this.
        has_eos = "<|im_end|>" in messages[-1]["content"][0]["text"]

        if has_eos:
            print("ENDED EARLY because of EOS token")

        return (hasattr(parsed, "answer") and parsed.answer is not None) or has_eos

    def _general_parse_key_value(self, text_blob: str) -> RawToolArgs:
        """Parses simple 'key: value' lines into a dictionary of strings."""
        args: RawToolArgs = {}
        # Handle potential empty input after stripping <tool> tags
        if not text_blob.strip():
            return args
        lines = text_blob.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or ": " not in line:
                # Skip empty lines or lines without the separator
                continue
            key, value_str = line.split(": ", 1)
            args[key.strip()] = value_str.strip()
        return args
