from mmte.models import BLIP2Chat


# Test Implementations
    
messages = [
    {
        "role": "user",
        "content": {"text": "Please describe this image.", \
                    "image_path": "/mnt/vepfs/zhangyichi/Trustworthy-MLLM/playground/closed-source/emma.jpg"},
    }
]

response = BLIP2Chat(model_id='pretrain_flant5xl').chat(messages)
print(response)