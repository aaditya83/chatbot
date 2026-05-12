# End-to-End AI Assistant

An all-in-one AI Assistant built with Streamlit and various machine learning tools. 

## Features

This application includes four main tabs:
1. **💬 Chatbot**: An interactive chatbot using `meta-llama/Meta-Llama-3-8B-Instruct` via the Hugging Face Inference API.
2. **😊 Sentiment Analyzer**: A machine learning model (trained on IMDB movie reviews) that predicts the sentiment of text (Positive, Negative, or Neutral).
3. **📝 Summarizer**: Summarizes long text into short, concise summaries (20-25 words) using the Llama-3 model.
4. **📄 PDF Q&A**: Upload a PDF document and ask questions about its content. It uses `SentenceTransformers` and `FAISS` for Retrieval-Augmented Generation (RAG).

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd chatbot
   ```

2. **Install the dependencies**:
   Make sure you have Python installed. Run the following command to install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Sentiment Model** (Optional, if `.pkl` files are missing):
   Ensure you have the `IMDB Dataset.csv` in the root folder, and run:
   ```bash
   python train_model.py
   ```
   This will generate `sentiment_model.pkl` and `vectorizer.pkl`.

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your browser, where you can navigate through the different tabs to interact with the various AI features.

## Files
- `app.py`: The main Streamlit dashboard.
- `ch.py`: Contains the chatbot logic calling the Hugging Face Inference API.
- `sent.py`: Handles sentiment analysis using the pre-trained Logistic Regression model.
- `sum.py`: Implements text summarization using the Hugging Face Inference API.
- `pdf.py`: Implements the PDF text extraction, embedding creation, and RAG logic.
- `train_model.py`: Script to train the sentiment analysis model using the IMDB dataset.
