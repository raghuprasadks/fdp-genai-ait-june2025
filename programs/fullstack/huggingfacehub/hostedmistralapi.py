import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="nebius",
    api_key="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",
)
"""
completion = client.chat.completions.create(
    model="mistralai/Mistral-Nemo-Instruct-2407",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)
print(completion.choices[0].message)
"""



def get_response(prompt):
    print(f"Prompt: {prompt}")
    completion = client.chat.completions.create(
        model="mistralai/Mistral-Nemo-Instruct-2407",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return completion.choices[0].message.content

