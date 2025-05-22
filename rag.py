import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def process_pdf(uploaded_file):
    # Guardar temporalmente el PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Cargar y dividir el documento
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # Obtener embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    # Calcular dimensión del embedding dinámicamente
    embedding_dim = len(embeddings.embed_query("test"))

    # Conectar a Qdrant local
    qdrant_client = QdrantClient(host="localhost", port=6333)

    # Crear colección si no existe
    collection_name = "example_collection"
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing_collections:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )

    # Usar Qdrant como vector store
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings
    )

    vector_store.add_documents(documents=all_splits)

    return docs, vector_store






