import os

from datasets import Dataset, DatasetDict
from PIL import Image
from tqdm import tqdm

from r1_vlm.datasets.aok_vqa.load_aokvqa import get_coco_path, load_aokvqa

# see the aokvqa github repo for more information: https://github.com/allenai/aokvqa on downloading the dataset
aokvqa_dir = '/millcreek/data/aokvqa'

# this is just the coco dataset
coco_dir = '/millcreek/data/academic/coco'

# dict_keys(['split', 'image_id', 'question_id', 'question', 'choices', 'correct_choice_idx', 'direct_answers', 'difficult_direct_answer', 'rationales'])
# dict_keys(['split', 'image_id', 'question_id', 'question', 'choices', 'correct_choice_idx', 'direct_answers', 'difficult_direct_answer', 'rationales'])
# dict_keys(['split', 'image_id', 'question_id', 'question', 'choices', 'difficult_direct_answer'])

splits = {}
for split in ['train', 'val']: #['train', 'val', 'test']: - waiting for test split of coco dataset to download
    dataset = load_aokvqa(aokvqa_dir, split)
    
    # list of dictionaries we will pass to HF Datasets
    example_list = []   
    for example in tqdm(dataset, desc=f"Processing {split} split"):
        image_path = get_coco_path(split, example['image_id'], coco_dir)
        assert os.path.exists(image_path), f"Image path {image_path} does not exist"
        
        # Open and immediately close the image after reading
        with Image.open(image_path) as image:
            image = image.copy()  # Create a copy that persists after closing
        
        image_id = example['image_id']
        question_id = example['question_id']
        question = example['question']
        choices = example['choices']
        difficult_direct_answer = example['difficult_direct_answer']
        
        # these fields aren't present for the test split
        correct_choice_idx = example.get('correct_choice_idx', None)
        direct_answers = example.get('direct_answers', None)
        rationales = example.get('rationales', None)
        
        example_dict = {
            "image_path": image_path,
            "image": image,
            "image_id": image_id,
            "question_id": question_id,
            "question": question,
            "choices": choices,
            "difficult_direct_answer": difficult_direct_answer,
            "correct_choice_idx": correct_choice_idx,
            "direct_answers": direct_answers,
            "rationales": rationales,
        }
        
        example_list.append(example_dict)
        
    
    splits[split] = Dataset.from_list(example_list)

dataset = DatasetDict(splits)

# save the dataset to disk because pushing to hub is slow
save_path = "/millcreek/data/aokvqa/aokvqa_hf"
assert os.path.exists(save_path), f"Save path {save_path} does not exist"
dataset.save_to_disk(save_path)
