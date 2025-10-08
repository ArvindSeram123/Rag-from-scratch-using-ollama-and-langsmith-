
# 🧠 LangChain + Ollama + Chroma | RAG Example

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline using  
**LangChain**, **Ollama**, **Chroma**, and **SentenceTransformers**.  
It allows a local LLM to answer questions **based only on your documents**.

---

## 🚀 Overview

1. Load a text (e.g., a speech or document)
2. Split it into smaller chunks
3. Convert chunks into embeddings
4. Store embeddings in a Chroma vector database
5. Retrieve relevant chunks based on a query
6. Pass retrieved context + question to an Ollama LLM
7. Generate an accurate, context-aware response

---

## 🧩 Code

```python
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# -----------------------------
# Step 1: Load your text
# -----------------------------
speech = """Even though you are only a very small speck of ocean...
(put your full text here)
"""

# -----------------------------
# Step 2: Convert to Document
# -----------------------------
docs = [Document(page_content=speech)]

# -----------------------------
# Step 3: Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# -----------------------------
# Step 4: Create embeddings
# -----------------------------
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------
# Step 5: Store in vector DB
# -----------------------------
vectorstore = Chroma.from_documents(chunks, embedding)

# -----------------------------
# Step 6: Create retriever
# -----------------------------
retriever = vectorstore.as_retriever()

# -----------------------------
# Step 7: Define prompt template
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer based only on the context."),
    ("user", "Question: {question}\nContext: {context}")
])

# -----------------------------
# Step 8: Load Ollama model
# -----------------------------
model = ChatOllama(model="gpt-oss:120b-cloud")

# -----------------------------
# Step 9: Parser for output
# -----------------------------
parser = StrOutputParser()

# -----------------------------
# Step 10: Build the chain
# -----------------------------
chain = prompt | model | parser

# -----------------------------
# Step 11: Direct LLM call (no retrieval)
# -----------------------------
question = "What are the key messages of the speech?"
context = """Even though you are only a very small speck of ocean...
(put full speech here)"""

print(chain.invoke({"question": question, "context": context}))

# -----------------------------
# Step 12: Retrieval (RAG) mode
# -----------------------------
query = "What are the key messages of the speech?"
docs_context = retriever.get_relevant_documents(query)

# Concatenate top chunks as context
rag_context = "\n".join([d.page_content for d in docs_context])

# Generate final answer
print(chain.invoke({"question": query, "context": rag_context}))



