import json
from pathlib import Path
from typing import Dict, List
from tqdm.auto import tqdm
import torch

def load_json_data(json_path: str) -> Dict:

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_training_examples(data: Dict) -> List[Dict]:
    
    examples = []
    albums = data['albums'].values()
    
    with tqdm(total=sum(len(album['songs']) for album in albums), 
              desc="Preparing training examples") as pbar:
        for album in albums:
            for song in album['songs']:
                examples.append({
                    'text': f"Album: {album['title']}\nSong: {song['title']}\nLyrics: {song['lyrics']}"
                })
                pbar.update(1)
    
    return examples

def verify_cuda_support():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\nFound {gpu_count} CUDA device(s):")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    CUDA Capability: {props.major}.{props.minor}")
    else:
        print("\nNo CUDA devices found. Running on CPU.")