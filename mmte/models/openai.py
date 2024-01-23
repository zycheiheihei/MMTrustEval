from typing import List, Dict, Any
import openai
import yaml
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
import os
import base64


class OpenAIChat(BaseChat):
    """
    Chat class for OpenAI models, e.g. gpt-4-vision-preview
    """
    
    MODEL_CONFIG = {"gpt-4-vision-preview": 'configs/models/openai/openai.yaml',
                    "gpt-4-1106-preview": 'configs/models/openai/openai.yaml',
                    "gpt-3.5-turbo": 'configs/models/openai/openai.yaml'}
    
    def __init__(self, model_id='gpt-4v', use_proxy=True, **kargs) -> None:
        super().__init__(**kargs)
        self.model_id = model_id
        self.model_type = self.id_to_type(model_id)
        self.model_family = 'gpt'
        config = self.MODEL_CONFIG[self.model_type]
        with open(get_abs_path(config)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        if use_proxy:
            os.environ['http_proxy'] = self.model_config.get('proxy')
            os.environ['https_proxy'] = self.model_config.get('proxy')
        self.api_key = self.model_config.get('api_key')
        self.max_retries = self.model_config.get('max_retries', 10)
        self.timeout = self.model_config.get('timeout', 5)
        self.client = openai.OpenAI(api_key=self.api_key, max_retries=self.max_retries, timeout=self.timeout)
        
        
        
        
    def chat(self, messages: List, **generation_kwargs):
        conversation = []
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if isinstance(message['content'], dict):
                    if len(conversation)==0 and message["role"] == "system":
                        raise AttributeError("Currently OpenAI doesn't support images in the first system message but this may change in the future.")
                        
                    # multimodal content
                    text = message['content']['text']
                    image_path = message['content']['image_path']
                    local_image = os.path.exists(image_path)
                    content = [
                        {
                            "type": "text",
                            "text": text,
                        },
                        {
                            "type": "image_url",
                            "image_url": { "url":  f"data:image/jpeg;base64, {self.encode_image(image_path)}" if local_image else image_path}
                        }
                    ]
                else:
                    content = message['content']
                conversation.append({"role": message["role"], "content": content})
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
        
        # Create Completion Request
        raw_request: Dict[str, Any] = {
            "model": self.model_type,
            "messages": conversation,
        }
        
        # Generation Configuration
        raw_request["temperature"] = generation_kwargs.get("temperature", 1.0)
        raw_request["max_tokens"] = generation_kwargs.get("max_tokens", 100)
        raw_request["n"] = generation_kwargs.get("num_completions", 1)
        if "stop_sequences" in generation_kwargs:
            raw_request["stop"] = generation_kwargs.get("stop_sequences")
        if "do_sample" in generation_kwargs and not generation_kwargs.get("do_sample"):
            raw_request["temperature"] = 0.0
        if "logprobs" in generation_kwargs and "vision" not in self.model_type:
            raw_request["logprobs"] = generation_kwargs.get("logprobs", False)
            
        response = self.client.chat.completions.create(**raw_request)
        
        response_message = response['choices'][0]['message']['content']
        finish_reason = response['choices'][0]['finish_reason']
        logprobs = response['choices'][0]['logprobs']
        
        return Response(self.model_id, response_message, logprobs, finish_reason)

    
    @classmethod
    def id_to_type(cls, id: str) -> str:
        if 'v' in id.strip("preview"):
            return "gpt-4-vision-preview"
        elif 'gpt-4' in id:
            return "gpt-4-1106-preview"
        else:
            return "gpt-3.5-turbo"
    
    
    # Function to encode the image
    @classmethod
    def encode_image(cls, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    
    
