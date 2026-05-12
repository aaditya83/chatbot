from huggingface_hub import InferenceClient
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant. Answer clearly and concisely."}
]

def ch(user_input):
    messages.append({"role": "user", "content": user_input})
    
    stream = client.chat.completions.create(
        messages=messages,
        stream=True,
        max_tokens=200,
    )

    bot_response = ""
    for message in stream:
        if not message.choices:
            continue
        delta = message.choices[0].delta
        if delta and delta.content:
            bot_response += delta.content
            
    messages.append({"role": "assistant", "content": bot_response})
    return bot_response

if __name__ == "__main__":
    print("Type 'quit' or 'exit' to end the conversation.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        print("Bot:", ch(user_input), "\n")