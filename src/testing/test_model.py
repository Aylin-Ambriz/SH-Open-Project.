import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import os

class SHModelTester:
    def __init__(self, model_path="models/sound_horizon_final"):
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
       
        if not self.model_path.exists():
            raise ValueError(f"Model path '{model_path}' does not exist.")
        
        print("Loading model and tokenizer...")
        try:
         
            print("Available files in model directory:")
            for file in self.model_path.glob("*"):
                print(f"- {file.name}")
            
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=True
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=True
            )
            
           
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Model and tokenizer loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("\nModel directory contents:")
            for root, dirs, files in os.walk(self.model_path):
                for file in files:
                    print(f"- {os.path.join(root, file)}")
            raise
    
    def generate_text(self, prompt, 
                     max_length=200,
                     temperature=0.7,
                     top_k=50,
                     top_p=0.9,
                     num_return_sequences=1,
                     do_sample=True,
                     **kwargs):
        """
        Generate text with various parameters:
        - prompt: Input text to generate from
        - max_length: Maximum length of generated text (including prompt)
        - temperature: Higher = more creative, lower = more focused
        - top_k: Number of highest probability tokens to consider
        - top_p: Cumulative probability threshold for tokens
        - num_return_sequences: Number of different outputs to generate
        - do_sample: Whether to use sampling (True) or greedy decoding (False)
        """
        try:
        
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
           
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
            
           
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            return generated_texts
            
        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            return []
    
    def evaluate_coherence(self, text):
        """
        Basic evaluation of text coherence focused on Sound Horizon-specific elements
        """
        try:
            scores = {
                'length': len(text.split()),
                'album_mention': 'Album:' in text,
                'song_mention': 'Song:' in text,
                'lyrics_mention': 'Lyrics:' in text,
                'has_linebreaks': '\n' in text,
                'avg_line_length': len(text) / (text.count('\n') + 1) if text else 0
            }
            return scores
        except Exception as e:
            print(f"Error during coherence evaluation: {str(e)}")
            return {}

def main():
    try:
 
        print("Initializing model tester...")
        tester = SHModelTester()
        
        test_prompts = [
            "Album: Roman\nSong: ",
            "Album: Märchen\nSong: ",
            "Album: Moira\nSong: ",
            "Album: ",
            "Interview: ",
        ]
        
        print("\n=== Testing Different Generation Parameters ===")
        
      
        print("\n1. Testing temperature effects (creativity vs coherence):")
        prompt = "Album: Roman\nSong: "
        for temp in [0.3, 0.7, 1.0]:
            print(f"\nTemperature = {temp}:")
            try:
                texts = tester.generate_text(prompt, temperature=temp)
                for idx, text in enumerate(texts):
                    print(f"Generation {idx + 1}:")
                    print(text[:200] + "..." if len(text) > 200 else text)
            except Exception as e:
                print(f"Error at temperature {temp}: {str(e)}")
        
        
        print("\n2. Testing different prompt types:")
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            try:
                texts = tester.generate_text(prompt)
                print(texts[0][:200] + "..." if len(texts[0]) > 200 else texts[0])
                scores = tester.evaluate_coherence(texts[0])
                print(f"Evaluation scores: {scores}")
            except Exception as e:
                print(f"Error with prompt '{prompt}': {str(e)}")
        
        print("\n3. Generating multiple versions of the same prompt:")
        prompt = "Album: Märchen\nSong: "
        try:
            texts = tester.generate_text(prompt, num_return_sequences=3)
            for i, text in enumerate(texts, 1):
                print(f"\nVersion {i}:")
                print(text[:200] + "..." if len(text) > 200 else text)
        except Exception as e:
            print(f"Error generating multiple versions: {str(e)}")
    
        print("\n4. Testing long-form generation:")
        try:
            long_prompt = "Album: Roman\nSong: Yield\nLyrics: "
            texts = tester.generate_text(
                long_prompt,
                max_length=400,
                temperature=0.8,
                top_p=0.95
            )
            print(texts[0])
        except Exception as e:
            print(f"Error in long-form generation: {str(e)}")
            
    except Exception as e:
        print(f"Critical error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()