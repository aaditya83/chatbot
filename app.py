import streamlit as st
from ch import ch
from sent import sent
from sum import sum
from pdf import pdf

st.set_page_config(page_title="AI Assistant", layout="wide")

st.title("🚀 End-to-End AI Assistant")

tabs = st.tabs(["💬 Chatbot", "😊 Sentiment", "📝 Summarizer", "📄 PDF Q&A"])


with tabs[0]:
    st.header("Chatbot")

    user_question = st.text_input("Ask me anything:")

    if st.button("Get Response", key="chat"):
        if user_question:
            response = ch(user_question)
            st.success(response)



with tabs[1]:
    st.header("Sentiment Analyzer")

    text = st.text_area("Enter text for sentiment analysis:")

    if st.button("Analyze Sentiment", key="sentiment"):
        if text:
            sentiment = sent(text)
            st.success(f"Sentiment: {sentiment}")



with tabs[2]:
    st.header("Text Summarizer")

    long_text = st.text_area("Paste long text to summarize:")

    if st.button("Summarize", key="summary"):
        if long_text:
            summary = sum(long_text)
            st.success(summary)


            
with tabs[3]:
    st.header("PDF Question Answering")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    question = st.text_input("Ask a question from the PDF:")

    if st.button("Get Answer", key="pdf"):
        if uploaded_file and question:
            answer = pdf(uploaded_file, question)
            st.success(answer)