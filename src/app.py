# Librerias
import streamlit as st
import ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from memoria import get_by_session_id
from rag import process_pdf
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()
tracing = os.environ.get("LANGSMITH_TRACING", "false")
api_key = os.environ.get("LANGSMITH_API_KEY")

# Sidebar de selección de modelo
def list_models():
    models_running = ollama.list()['models']
    available_models = [model["model"] for model in models_running]
    return available_models

lista = list_models()

if 'model_selection' not in st.session_state:
    st.session_state.model_selection = lista[0] if lista else None

# Configuración avanzada en el Sidebar
with st.sidebar:
    st.title('🤖🧠 Opciones')
    st.session_state.model_selection = st.selectbox(
        'Selecciona el modelo',
        options=lista,
        index=lista.index(st.session_state.model_selection) if st.session_state.model_selection in lista else 0
    )
    
    # Parámetros
    st.subheader("Configuración avanzada")
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.1, 0.01)
    top_p = st.slider("Top-p:", 0.0, 1.0, 0.9, 0.01)
    top_k = st.slider("Top-k:", 1, 100, 50, 1)

# Interfaz principal
st.title("Chatbot RAG")
st.write("Cargar archivo PDF para procesar:")
uploaded_file = st.file_uploader("Selecciona un PDF", type="pdf")

if uploaded_file:
    # Procesamiento del PDF
    docs, vector_store = process_pdf(uploaded_file)
    st.success(f"Documento cargado correctamente. Páginas: {len(docs)}")

    # Visualización del contenido inicial
    st.write(f"**Contenido de la primera página:**\n")
    st.write(f"{docs[0].page_content[:200]}...\n")
    st.write(f"**Metadatos:**\n", docs[0].metadata)

    # Chatbot con memoria
    st.write("Realiza una consulta:")
    query = st.text_input("Escribe tu pregunta:")

    if query:
        # Almacenamiento del historial
        history = get_by_session_id("1")
        
        # Prompt del modelo
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're an assistant who answers questions based on the document."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

       
        chain = prompt | ChatOllama(
            model=st.session_state.model_selection, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k
        )

        # Enlace con memoria
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Realizar la consulta
        response = chain_with_history.invoke({"question": query}, config={"configurable": {"session_id": "1"}})

        # Mostrar respuesta
        st.write("**Respuesta del modelo:**")
        st.write(response)

        # Mostrar historial
        st.write("**Historial del Chat:**")
        for msg in history.messages:
            st.write(msg)


