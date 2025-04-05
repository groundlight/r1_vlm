import numpy as np
import PIL
from datasets import Dataset
from tqdm import tqdm
from r1_vlm.tools.utils.image_hash_table import ImageHashTable
from typing import TypedDict



class DigitData(TypedDict):
    label: list[int]
    total: int

class DigitsAnswerTool:
    '''
    Uses ImageHashTable to store and lookup digit recognition and addition results.
    '''
    def __init__(self, dataset: Dataset):
        self.hash_table = ImageHashTable()
    
    def build_hash_table(self, dataset: Dataset) -> None:
        for example in tqdm(dataset, desc="Building hash table"):
            image = example["image"]
            label = example["label"]
            total = example["total"]
            
            assert isinstance(image, PIL.Image.Image)
            assert isinstance(label, list)
            assert isinstance(total, int)
            
            self.add_image(image, label, total)
    
    def add_image(self, image: PIL.Image.Image, data: dict):
        '''
        Adds image to the hash table with its label and total.
        '''
        data: DigitData = {"label": label, "total": total}
        self.hash_table.add_image(image, data)
    
    def lookup_image(self, image: PIL.Image.Image) -> DigitData:
        '''
        Looks up the image in the hash table and returns the label and total.
        '''
        return self.hash_table.lookup_image(image)

# Make DigitsAnswerTool accessible to get_answer
_digits_answer_tool = None

def set_digits_answer_tool(tool: DigitsAnswerTool):
    global _digits_answer_tool
    _digits_answer_tool = tool

def get_answer(image_name: str, task: str, **kwargs) -> str:
    '''
    Returns the answer, either listing or summing the digits in the given image.
    
    Args:
        image_name: str, the name of the image to submit to the tool.
        task: str, either "recognition" or "addition"
        kwargs: dict, should not be used. 
    
    
    Returns:
        A string representation of the answer. e.g. "[4, 7]" for recognition, or "11" for addition
        

    Examples:
        <tool>{"name": "get_answer", "args": {"image_name": "input_image", "task": "recognition"}}</tool> 
        <tool>{"name": "get_answer", "args": {"image_name": "input_image", "task": "addition"}}</tool>
    '''
    
    # NOTE: The tool cheats (purposely)! It's not a "serious" tool, but rather a proof of concept to verify tool calling works properly. 
    if _digits_answer_tool is None:
        print("Error: DigitsAnswerTool not initialized. Call set_digits_answer_tool first.")
        raise ValueError("DigitsAnswerTool not initialized. Call set_digits_answer_tool first.")
    
    valid_tasks = ["recognition", "addition"]
    if task not in valid_tasks:
        raise ValueError(f" Error: Invalid task: {task}. Valid tasks are: {valid_tasks}")

    images = kwargs["images"]
    
    image_to_use = images.get(image_name, None)
    
    if image_to_use is None:
        raise ValueError(f"Error: Image {image_name} not found. Valid image names are: {images.keys()}")
    
    result = _digits_answer_tool.lookup_image(image_to_use)
    
    if task == "recognition":
        data = result["label"]
    else:
        data = result["total"]
        
    return str(data)

    
    
