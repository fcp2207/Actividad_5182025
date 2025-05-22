import streamlit as st
import ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from src.memoria import get_by_session_id
from rag import process_pdf
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()
tracing = os.environ.get("LANGSMITH_TRACING", "false")
api_key = os.environ.get("LANGSMITH_API_KEY")

# Función para listar modelos activos en Ollama
def list_models():
    models_running = ollama.list()['models']
    available_models = [model["model"] for model in models_running]
    return available_models

# UI - Configuración lateral
lista = list_models()
if 'model_selection' not in st.session_state:
    st.session_state.model_selection = lista[0] if lista else None

with st.sidebar:
    st.title('🤖🧠 Opciones')
    st.session_state.model_selection = st.selectbox(
        'Selecciona el modelo',
        options=lista,
        index=lista.index(st.session_state.model_selection) if st.session_state.model_selection in lista else 0
    )
    st.subheader("Configuración avanzada")
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.1, 0.01)
    top_p = st.slider("Top-p:", 0.0, 1.0, 0.9, 0.01)
    top_k = st.slider("Top-k:", 1, 100, 50, 1)

# UI - Pantalla principal
st.title("Chatbot RAG")
st.write("Cargar archivo PDF:")
uploaded_file = st.file_uploader("Selecciona un PDF", type="pdf")

if uploaded_file:
    docs, vector_store = process_pdf(uploaded_file)
    st.success(f"Documento cargado correctamente. Páginas: {len(docs)}")

    st.write(f"**Contenido de la primera página:**\n")
    st.write(f"{docs[0].page_content[:200]}...\n")
    st.write(f"**Metadatos:**\n", docs[0].metadata)

    st.write("Realiza una consulta:")
    query = st.text_input("Escriba su pregunta:")

    if query:
        # Recuperar historial
        history = get_by_session_id("1")

        # Buscar contexto relevante con RAG
        retrieved_docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Crear prompt con contexto
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Usa la siguiente información del documento para responder la pregunta:\n\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        # Construir cadena con historial
        chain = prompt | ChatOllama(
            model=st.session_state.model_selection,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Ejecutar cadena
        response = chain_with_history.invoke(
            {"question": query, "context": context},
            config={"configurable": {"session_id": "1"}}
        )

        st.write("**Respuesta del modelo:**")
        st.write(response)

        st.write("**Historial del Chat:**")
        for msg in history.messages:
            st.write(msg)



