🤖 RAG-based AI Chatbot (LLM Model)
A high-performance, context-aware AI Chatbot built using Retrieval-Augmented Generation (RAG) architecture. This bot leverages Hugging Face models to retrieve relevant information from a knowledge base and provide accurate, grounded responses.

Live Demo: https://ai-chatbot-llm-model-1.onrender.com

🚀 Features
Contextual Awareness: Uses RAG to fetch relevant data before generating answers, reducing hallucinations.

Hugging Face Integration: Powered by state-of-the-art open-source LLMs hosted/integrated via Hugging Face.

Vector Search: Implements efficient document retrieval to provide specific answers based on provided data.

Real-time Interaction: Optimized for low-latency conversations, deployed on the Render platform.

🛠️ Tech Stack
Language: Python

Frontend: Streamlit

AI Framework: LangChain / LangGraph (for orchestration)

Model Provider: Hugging Face (Inference Endpoints / Hub)

Vector Store: ChromaDB / FAISS (for high-speed similarity search)

Deployment: Render

📂 Project Structure
Plaintext
├── app.py              # Main Streamlit application
├── core/               # RAG logic, embeddings, and LLM chain configurations
├── requirements.txt    # Production-ready dependencies
├── runtime.txt         # Python environment version (3.11)
└── data/               # Documents used for the knowledge base
⚙️ How It Works
Ingestion: Documents are broken into chunks and converted into embeddings.

Retrieval: When a user asks a question, the system searches the Vector Store for the most relevant chunks.

Generation: The Hugging Face model receives the question + the retrieved context to generate a precise answer.

Developed by Anmol Panwar

Aspiring Machine Learning Engineer | BCA Graduate
