import re
import pickle
import numpy as np
import os

def cleancode(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


def sent(text):
    text = cleancode(text)
    text_vec = vectorizer.transform([text])

    probs = model.predict_proba(text_vec)[0]
    pred = model.predict(text_vec)[0]

    confidence = np.max(probs)

    if confidence < 0.55:
        return "neutral"
    return pred


if __name__ == "__main__":
    while True:
        i = input("Enter your review: ")
        if i.lower() == "exit":
            break
        print(sent(i))