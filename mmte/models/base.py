from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


class BaseChat(ABC):
    """
    Base class for models to be evaluated in a generative/chat manner.
    """
    
    model_id = ''   # ID for a chat model, e.g., minigpt-4-vicuna-7b-v0
    model_family = '' # Family of the model, e.g., minigpt-4
    model_type = '' # Type to load specific configuration for a certain type of model, e.g., vicuna-7b-v0
    
    
    def __init__(self, **kargs) -> None:
        pass
    

    @classmethod
    def id_to_type(cls, id:str) -> str:
        """
        Convert model_id to model_type in order to locate the model config
        """
        return ''
    
    @abstractmethod
    def chat(self, 
             messages: List, 
             **generation_kwargs,
             ):
        """
        Chat interface for generative evaluation with batch size of 1.
        
        messages: a list of messages, comprising the conversation history and following the format 
            [
                {
                    'role': 'system'/'user'/'assistant', 
                    'content': str/dict
                },
                ...
            ], 
            where content is a dict {'text': str, 'image_path': str} when it's multimodal.
        generation_kwargs: generation configuration specified for different models, including "temperature", "do_sample", "max_tokens", "stop_sequences", "logprobs", etc.
            
            
        """
        raise NotImplementedError
    



@dataclass
class Response:
    
    model_id: str
    # The identifier of the model giving the response

    content: str
    # The content of the response
    
    logprobs: Any
    # The log probabilities of the output tokens
    
    finish_reason: Optional[str]
    

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Response":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, item):
        return getattr(self, item)
    

    
    