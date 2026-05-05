from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_vectorstore_from_uploads(uploaded_files):
    documents = []

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for uploaded_file in uploaded_files:
            pdf_path = temp_path / uploaded_file.name
            pdf_path.write_bytes(uploaded_file.getvalue())

            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())

    if not documents:
        raise ValueError("No readable text found in the uploaded PDF files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("PDF loaded, but no text chunks could be created.")

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore
