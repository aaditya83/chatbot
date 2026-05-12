import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

def cleancode(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("IMDB Dataset.csv")
    
    print("Cleaning text...")
    df['cleaned_review'] = df['review'].apply(cleancode)

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000)
    x = vectorizer.fit_transform(df['cleaned_review']) # Kept as sparse matrix!
    y = df["sentiment"]

    print("Splitting dataset...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = LogisticRegression()
    model.fit(x_train, y_train)

    print("Saving model and vectorizer...")
    with open("sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Done! Model saved to sentiment_model.pkl and vectorizer.pkl.")
