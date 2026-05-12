import re
import pickle
import numpy as np

def cleancode(text):
    text=text.lower()
    text=re.sub(r"[^a-z0-9\s]", " ",text)
    return text

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def sent(text):
    text=cleancode(text)
    text_vec=vectorizer.transform([text])
    probs = model.predict_proba(text_vec)[0]
    if 0.40 <= probs[0] <= 0.60:
        return "neutral"
    else:
        return model.predict(text_vec)[0]
    
if __name__ == "__main__":
    while True:
        i = input("Enter your review: ")
        if i.lower() == "exit":
            break
        else:
            print(sent(i))
    