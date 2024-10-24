from typing import List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LanguageModel:
    """Wrapper for large language model interactions"""
    
    def __init__(self, 
                 model_name: str = "google/gemini-1.5-flash",
                 max_length: int = 1_000_000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_length = max_length
        self.model.eval()
        
    def generate(self, 
                prompt: str,
                max_new_tokens: int = 100,
                temperature: float = 0.7,
                do_sample: bool = True,
                num_return_sequences: int = 1) -> List[str]:
        """Generate text from prompt"""
        inputs = self.tokenize(prompt, return_tensors=True)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
    
    def tokenize(self, 
                text: Union[str, List[str]], 
                return_tensors: bool = False) -> Union[List[int], dict]:
        """Tokenize input text"""
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True if return_tensors else False,
            return_tensors="pt" if return_tensors else None
        )
        return tokens
    
    def get_token_count(self, text: str) -> int:
        """Get token count for text"""
        return len(self.tokenize(text))
    
    def format_system_prompt(self, content: str) -> str:
        """Format system prompt based on model requirements"""
        return f"<|system|>{content}</s>"
    
    def format_user_prompt(self, content: str) -> str:
        """Format user prompt based on model requirements"""
        return f"<|user|>{content}</s>"
    
    def format_assistant_prompt(self, content: str) -> str:
        """Format assistant prompt based on model requirements"""
        return f"<|assistant|>{content}</s>"