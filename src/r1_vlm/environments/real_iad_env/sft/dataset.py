import json
import os

from datasets import Dataset
from tqdm import tqdm

# imports here
from r1_vlm.environments.real_iad_env.completion_generation import (
    hash_example,
    setup_env,
    setup_model_and_processor,
)


def load_completion_dataset(path:str = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/real_iad_env/completion_results/completion_results.jsonl"):
    '''
    Loads the completion dataset from the given path. Returns a dictionary of examples, mapping the image hash to the highest scoring completion for that image.
    '''
    
    data = []
    with open(path, "r") as f:
        for line in tqdm(f):
            data.append(json.loads(line.strip()))
    
    # maps the image hash to the examples that have that hash to organize the data
    hash_to_examples = {}
    
    for example in data:
        hash_value = example["image_hash"]
        
        if hash_value not in hash_to_examples:
            hash_to_examples[hash_value] = []
        
        hash_to_examples[hash_value].append(example)
    
    
    filtered_data = {}
        
    # only keep the top scoring completion for each image
    for hash_value, examples in hash_to_examples.items():
        examples.sort(key=lambda x: x["reward"], reverse=True)
        filtered_data[hash_value] = examples[0]

    return filtered_data


def create_sft_dataset():
    '''
    Returns a dataset of SFT examples for finetuning the model to generate completions that backtrack.
    '''
    completion_dataset = load_completion_dataset()

    _, processor = setup_model_and_processor(checkpoint_path="/millcreek/home/sunil/r1_vlm/vlm-r1-real-iad-simple-env-budget-forcing-longer-ignore-strings/checkpoint-100")

    env = setup_env(processor=processor)
    train_dataset, _ = env.get_dataset()
    
    # convert the training dataset to a dictionary mapping the image hash to the example so we can look up each example by image hash
    train_dataset_hash_to_example = {}
    for example in tqdm(train_dataset, desc="Hashing training dataset"):
        hash_value = hash_example(example)
        train_dataset_hash_to_example[hash_value] = example
    
    
    sft_examples = []
    for example in completion_dataset.values():
        hash_value = example["image_hash"]
        train_example = train_dataset_hash_to_example[hash_value]
        
        messages = train_example["messages"]
        
        # remove nonsense keys added by huggingface
        for message in messages:
                content = message["content"]
                message["content"] = [
                    {k: v for k, v in item.items() if v is not None} for item in content
                ]
        
        # add the assistant's completion to the messages
        completion_message = example["completion_messages"][0]
        
        # if the completion message ends in <|im_end|> then remove it
        if completion_message["content"][0]["text"].endswith("<|im_end|>"):
            completion_message["content"][0]["text"] = completion_message["content"][0]["text"].rstrip("<|im_end|>")
        
        train_example["messages"].append(completion_message)
        
        sft_examples.append({"messages": train_example["messages"]})
    
    return sft_examples


def save_as_jsonl(dataset: Dataset, path: str):
    '''
    Saves the dataset as a jsonl file and image files in the ms swift format for finetuning.
    
    path: the directory to save the dataset to. This function will create a subdirectory called "dataset" with a jsonl file and a subdirectory called "images" with the image files.
    '''
    
    # create dataset directory
    dataset_dir = os.path.join(path, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # create images directory
    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # create a jsonl file to save the dataset
    jsonl_file = os.path.join(dataset_dir, "dataset.jsonl")
    
    example_count = 0
    
    for example in tqdm(dataset, desc="Saving dataset"):
        messages = example["messages"]
        
        # remove the image content from the messages to save separately and replace the adjacent text with the <image> placeholder for ms swift to handle
        for element in messages:
            if element["role"] == "user":
                for index in range(len(element["content"])):
                    if element["content"][index]["type"] == "image":
                        # grab the image
                        image = element["content"][index]["image"]
                        
                        # add <image> placeholder to the text right after the image
                        element["content"][index-1]["text"] = "<image>" + element["content"][index-1]["text"]
                        
                        # remove the image from the content
                        element["content"].pop(index)
                        break
                    
        # so now we have the messages with the image content stored separately and the text content amended to include the <image> placeholder
        # now we need to save the image to disk
        image_path = os.path.join(images_dir, f"image_{example_count}.png")
        image.save(image_path)
        
        data = {
            "messages": messages,
            "images": [image_path]
        }
        
        
        # each content element in messages is currnetly a list of dicts that look like {"type": "text", "text": "..."}
        # ms swift just wants the raw text {"role": "user", "content": "..."}, so we need to flatten the content list
        for message in messages:
            # the content should be only one dict long, if it isn't then we need to raise an error
            if len(message["content"]) != 1:
                raise ValueError(f"Message content should be a list of one dict, but got {len(message['content'])}")
            
            # replace the content with just the text data
            message["content"] = message["content"][0]["text"]
            

        with open(jsonl_file, "a") as f:
            f.write(json.dumps(data) + "\n")
        
        example_count += 1
    

if __name__ == "__main__":
    dataset = create_sft_dataset()
    save_as_jsonl(dataset, "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/real_iad_env/sft/")