from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import InferenceClient

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def pdf(uploaded_file, question):
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
        
    chunks = chunk_text(full_text)
    if not chunks:
        return "Could not extract text from the PDF."
        
    chunk_embeddings = embedder.encode(chunks)
    
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))
    
    q_emb = embedder.encode([question])
    k = min(4, len(chunks))
    distances, indices = index.search(np.array(q_emb), k)
    context_chunks = [chunks[i] for i in indices[0]]
    
    context = "\n\n".join(context_chunks)

    messages = [
        {"role": "system", "content": "You answer questions using ONLY the provided PDF context."},
        {"role": "user", "content": f"Context from PDF:\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        messages=messages,
        stream=False,
        max_tokens=300,
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    print("Type 'quit' or 'exit' to end.\n")
    while True:
        question = input("You: ")
        if question.lower() in ["quit", "exit"]:
            break
        print("Bot:", pdf("document.pdf", question), "\n")
