from openai import OpenAI
import os

client = OpenAI(
    api_key="sk-bd435d0262f944cdb46fd6057c495011",
    # api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

completion = client.chat.completions.create(
    model="qwen-vl-max",  # Using qwen-vl-max as an example, you can change the model name as needed. Model list: https://www.alibabacloud.com/help/model-studio/getting-started/models
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                    },
                },
                {"type": "text", "text": "What scene is depicted in this image?"},
            ],
        },
    ],
)

print(completion.choices[0].message.content)