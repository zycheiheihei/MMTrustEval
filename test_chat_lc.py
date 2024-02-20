from mmte.models import QwenChat, MMICLChat, OtterChat


# Test Implementations
    
messages = [
    {
        "role": "user",
        "content": {"text": "What is in this image?", \
                    "image_path": "/data/zhangyichi/Trustworthy-MLLM/playground/Qwen-VL-hf/demo.jpeg"},
    }
]

response = OtterChat(model_id='otter-mpt-7b-chat').chat(messages)
print(response)