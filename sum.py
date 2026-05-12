from huggingface_hub import InferenceClient

client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")

SYSTEM_PROMPT = (
    "You are a professional text summarizer. "
    "Summarize input into 20–25 words. "
    "Keep only key meaning. Be precise and concise."
)

def sum(text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=60,
        stream=False
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print("AI Summarizer (type 'exit' to quit)\n")
    while True:
        text = input("Enter text: ")
        if text.lower() in ["exit", "quit"]:
            break
        summary = sum(text)
        print("Summary:", summary, "\n")