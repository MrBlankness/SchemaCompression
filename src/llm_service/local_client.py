import os
import re
import torch
import logging
import time
import random
from typing import List, Dict, Any, Union, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.core.interfaces import ILLMClient

logger = logging.getLogger(__name__)

class LocalLLMClient(ILLMClient):
    """
    Client for running local LLMs usage HuggingFace Transformers.
    """
    def __init__(self, model_path: str, device: str = None, max_retries: int = 3, seed: Optional[int] = None, enable_thinking: Optional[bool] = None, max_new_tokens: int = 1024):
        self.model_path = model_path
        self.max_retries = max_retries
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        
        # Determine device map logic
        self.device_map = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading local model from {model_path} with device_map='{self.device_map}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype="auto", 
                device_map=self.device_map
            )
            # Ensure we have a pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Record main device for manually moving tensors if needed (though accelerate handles this mostly)
            self.main_device = self.model.device
                
        except Exception as e:
            logger.critical(f"Failed to load local model: {e}")
            raise e

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.7, **kwargs) -> Tuple[str, int]:
        """
        Generates response using the local model.
        Returns: (response_text, input_token_count)
        """
        try:
            call_seed = kwargs.pop("seed", self.seed)
            if call_seed is not None:
                random.seed(call_seed)
                torch.manual_seed(call_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(call_seed)

            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    template_kwargs = {
                        "tokenize": False, 
                        "add_generation_prompt": True,
                    }
                    if self.enable_thinking is not None:
                        template_kwargs["enable_thinking"] = self.enable_thinking
                        
                    prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
                except TypeError:
                    # Fallback if tokenizer doesn't accept the kwarg
                    if self.enable_thinking is not None:
                        logger.warning("Tokenizer.apply_chat_template refused 'enable_thinking' argument. Ignoring.")
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
            else:
                prompt = ""
                for msg in messages:
                    role = msg.get("role", "").upper()
                    content = msg.get("content", "")
                    prompt += f"{role}: {content}\n"
                prompt += "ASSISTANT:"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.main_device)
            input_length = inputs.input_ids.shape[1]
            
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else 1.0, 
                "pad_token_id": self.tokenizer.pad_token_id
            }
            if temperature == 0:
                gen_kwargs["do_sample"] = False
                kwargs.pop('top_p', None)
                kwargs.pop('top_k', None)
                
            gen_kwargs.update(kwargs)

            if not gen_kwargs.get("do_sample", False):
                # Some models carry non-default top_p/top_k in generation_config,
                # which can still trigger warnings in greedy decoding. Explicitly
                # set greedy-safe defaults to silence invalid-flag warnings.
                gen_kwargs["temperature"] = 1.0
                gen_kwargs["top_p"] = 1.0
                gen_kwargs["top_k"] = 50
            
            try:
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                
                generated_tokens = outputs[0][input_length:]
                response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Post-processing: Remove <think> blocks if they exist
                if self.enable_thinking:
                    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                
                return response_text, input_length
            except Exception as e:
                logger.error(f"Local LLM generation error: {e}")
                return "", input_length

        except Exception as e:
            logger.error(f"Local LLM unexpected error: {e}")
            return "", 0
            
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))