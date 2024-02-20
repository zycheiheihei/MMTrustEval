from typing import List
import torch
from mmte.models.base import BaseChat, Response
from PIL import Image
from transformers import CLIPImageProcessor
from mmte.models.otter import OtterForConditionalGeneration


class OtterChat(BaseChat):
    """
    Chat class for Otter models
    """
    
    MODEL_CONFIG = {"otter-mpt-7b-chat": "luodian/OTTER-Image-MPT7B",}
    
    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'otter'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        self.device = device
        config = self.MODEL_CONFIG[self.model_id]
        kwargs = {"device_map": self.device, "torch_dtype": torch.bfloat16}
        self.model = OtterForConditionalGeneration.from_pretrained(config, **kwargs)
        self.image_processor = CLIPImageProcessor()
        self.tokenizer = self.model.text_tokenizer
        self.tokenizer.padding_side = "left"
        self.model.to(self.device)
        self.model.eval()

        
    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        image = Image.open(image_path)
                        vision_x = self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0).to(self.device)
                        lang_x = self.tokenizer(
                            [self.get_formatted_prompt(user_message, no_image_flag=False),],
                            return_tensors="pt",
                        )
                        model_dtype = next(self.model.parameters()).dtype
                        vision_x = vision_x.to(dtype=model_dtype)
                    else:
                        # TODO: text only conversation
                        pass
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
            
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response = self.tokenizer.decode(generated_text[0]).split("<answer>")[-1].strip().replace("<|endofchunk|>", "")

        return Response(self.model_id, response, None, None)
    
    def get_formatted_prompt(self, question: str, no_image_flag: str) -> str:
        if no_image_flag:
            return f"User:{question} GPT:<answer>"
        else:
            return f"<image>User:{question} GPT:<answer>"
