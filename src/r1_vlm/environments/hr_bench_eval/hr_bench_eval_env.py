import json
import re
from typing import Any, Callable

from PIL import Image
from transformers import AutoProcessor
from datasets import Dataset
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
from r1_vlm.environments.tool_vision_env import ToolArgParser, ToolVisionEnv
from r1_vlm.datasets.hr_bench.hr_bench_base_for_eval import create_r1_hr_bench_simple_dataset
from r1_vlm.datasets.hr_bench.hr_bench_tool_use_r1 import create_r1_hr_bench_tool_use_dataset
from r1_vlm.tools.tool_prompts import SINGLE_TOOL_PROMPT_TEMPLATE_SIMPLIFIED
from r1_vlm.tools.zoom import parse_zoom_args, zoom


class SimpleHRBenchEvalEnv(SimpleVisionEnv):
    def __init__(
        self,
        split: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split = split
        self.parser = XMLParser(fields=["think", "answer"])
        self._fields = [
            ("think", ["think"]),
            ("answer", ["answer"]),
        ]

    def parse(self, text: str, strip: bool = True):
        return self.parser.parse(text, strip=strip)

    def get_dataset(self, max_size: int | None = None) -> Dataset:
        dataset = create_r1_hr_bench_simple_dataset(
            split=self.split,
            max_size=max_size,
        )
        dataset = preprocess_r1_dataset(dataset)
        return dataset

    def get_rubric(self):
        raise NotImplementedError(
            "get_rubric is not implemented for SimpleVStarEvalEnv because the env is used for evaluation only"
        )

class HRBenchToolEnv(ToolVisionEnv):
    def __init__(
        self,
        processing_class: AutoProcessor,
        split: str,
        tools_with_parsers: list[tuple[Callable, ToolArgParser]] = [
            (zoom, parse_zoom_args),
        ],
        max_steps: int = 3,
        tool_prompt_template: str = SINGLE_TOOL_PROMPT_TEMPLATE_SIMPLIFIED,
    ):
        super().__init__(
            processing_class=processing_class,
            tools_with_parsers=tools_with_parsers,
            max_steps=max_steps,
            tool_prompt_template=tool_prompt_template,
        )

        self.split = split
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
        max_size: int | None = None,
    ) -> Dataset:
        dataset = create_r1_hr_bench_tool_use_dataset(
            split=self.split,
            max_size=max_size,
        )
        dataset = self.inject_system_prompt(dataset)
        dataset = preprocess_r1_dataset(dataset)
        return dataset