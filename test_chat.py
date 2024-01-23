from mmte.models import OpenAIChat


# Test Implementations
    
messages = [
    {
        "role": "user",
        "content": {"text": "Describe this.", "image_path": "/mnt/vepfs/zhangyichi/Trustworthy-MLLM/playground/closed-source/emma.jpg"},
    }
]

OpenAIChat().chat(messages)