import os
import logging
import time
import random
import openai
import tiktoken
from openai import RateLimitError, APIConnectionError, APIStatusError
from typing import List, Dict, Any, Optional, Tuple
from src.core.interfaces import ILLMClient

logger = logging.getLogger(__name__)

class OpenAIClient(ILLMClient):
    """
    Adapter for OpenAI-compatible APIs (including official OpenAI, vLLM, DeepSeek, etc.).
    """
    def __init__(self, model_name=None, api_key=None, base_url=None, max_retries=3, seed: Optional[int] = None, max_new_tokens=1024, enable_thinking: Optional[bool] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
        self.max_retries = max_retries
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        
        if not self.api_key:
             self.api_key = "EMPTY" 
             
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 初始化本地编码器用于预计算 Token
        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # 如果是自定义模型名（如 deepseek），回退到 cl100k_base (GPT-4 家族通用)
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _estimate_prompt_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        本地估算 Prompt 的 Token 数量。
        参考 OpenAI 官方实现，消息体包含 role, content 以及额外的固定开销。
        """
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # 每条消息的固定开销 <im_start>...<im_end>
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":  # 如果有 name 字段，额外 +1
                    num_tokens += 1
        num_tokens += 2  # 助手回复前的固定开销
        return num_tokens

    def generate(self, messages, temperature=0.7, **kwargs) -> Tuple[str, int]:
        # 1. 无论请求成功与否，先预计算输入长度
        estimated_input_tokens = self._estimate_prompt_tokens(messages)
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_new_tokens,
        }
        if self.seed is not None and "seed" not in kwargs:
            params["seed"] = self.seed
        params.update(kwargs)

        if self.enable_thinking is not None:
            if "extra_body" not in params:
                params["extra_body"] = {}
            if "chat_template_kwargs" not in params["extra_body"]:
                params["extra_body"]["chat_template_kwargs"] = {}
            params["extra_body"]["chat_template_kwargs"]["thinking"] = self.enable_thinking

        # ---------------------------------------------------------
        # Backoff & Retry Logic
        # ---------------------------------------------------------
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**params)
                if response.choices:
                    content = response.choices[0].message.content or ""
                    # 优先使用 API 返回的真实 usage，如果缺失则使用预估值
                    usage = getattr(response, "usage", None)
                    actual_tokens = usage.prompt_tokens if usage else estimated_input_tokens
                    
                    time.sleep(0.2)
                    return content, actual_tokens
                
                # 如果有响应但没有 choices (罕见)
                return "", estimated_input_tokens
            
            except Exception as e:
                # Determine if we should retry
                should_retry = False
                error_msg = str(e).lower()
                
                # 限流或网络抖动
                if isinstance(e, RateLimitError) or isinstance(e, APIConnectionError):
                    should_retry = True
                elif "429" in error_msg or "rate limit" in error_msg:
                    should_retry = True
                elif isinstance(e, APIStatusError) and e.status_code >= 500:
                    should_retry = True

                if should_retry:
                    if attempt < self.max_retries:
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Retrying... ({e}).")
                        time.sleep(sleep_time)
                        continue
                    else:
                        logger.error(f"Max retries exceeded. Error: {e}")
                else:
                    # 关键修改：即使是 400 错误（如 Context Window Exceeded），也记录 Token 成本
                    logger.error(f"LLM generation failed (Non-retriable): {e}")
                    break
                    
        # 即使循环结束（所有重试失败），也返回预计算的 Token 数
        return "", estimated_input_tokens