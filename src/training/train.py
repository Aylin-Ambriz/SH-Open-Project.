import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from tqdm.auto import tqdm
import logging
from pathlib import Path

from ..preprocessing.dataProcessor import load_json_data, prepare_training_examples, verify_cuda_support
from ..preprocessing.jsonprocessor import SoundHorizonProcessor

def train():
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    

    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    config = {
        "output_dir": str(models_dir / "sound_horizon_model"),
        "model_name": "gpt2",
        "num_train_epochs": 3,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "warmup_steps": 100
    }
    
    verify_cuda_support()
    
    print("Loading data...")
    with tqdm(total=3, desc="Setup steps") as pbar:
      
        json_file = data_dir / "jsonprocessed.json"
        text_file = data_dir / "SH Discography.txt"
        
        if not json_file.exists():
            if not text_file.exists():
                raise FileNotFoundError(
                    f"Please place 'SH Discography.txt' in {data_dir}"
                )
            processor = SoundHorizonProcessor()
            processor.base_dir = data_dir
            processor.input_file = text_file
            processor.output_file = json_file
            processor.process()
        
     
        data = load_json_data(str(json_file))
        examples = prepare_training_examples(data)
        dataset = Dataset.from_list(examples)
        pbar.update(1)
        
        print("\nInitializing model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            
            results = tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors=None
            )
           
            results["labels"] = results["input_ids"].copy()
            return results
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        pbar.update(1)
        
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            device_map="auto",
            torch_dtype=torch.float32
        )
        model.resize_token_embeddings(len(tokenizer))
        
        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=16,
            learning_rate=config["learning_rate"],
            weight_decay=0.01,
            warmup_steps=config["warmup_steps"],
            fp16=False,
            bf16=False, 
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            logging_steps=10,
            save_strategy="steps",
            evaluation_strategy="no",
            save_steps=100,
            optim="adamw_torch",
            report_to="tensorboard",
            save_total_limit=2,
            logging_dir=str(models_dir / "logs"),  
            use_mps_device=False 
        )
        pbar.update(1)

    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=None
    )
    
    trainer.train()
    
    print("\nSaving model...")
    save_path = models_dir / "sound_horizon_final"
    trainer.save_model(save_path)
    print(f"Model saved to {save_path}")
    
    return trainer 

if __name__ == "__main__":
   
    Path("models").mkdir(exist_ok=True)
    
 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    
    trainer = train()