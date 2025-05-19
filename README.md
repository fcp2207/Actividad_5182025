CHATBOT-RAG

Este chatbot tiene como finalidad responder preguntas acerca de algun documento pdf que se le adjunte, funciona mediante la tecnica de RAG (Retrieval-Augmented-Generation) este proyecto integra:

- Langchain para la gestión de LLMs y generación de embeddings
- ChromaDB para almacenamiento de vetores
- Ollama para manejo del modelo de lenguaje
- Langsmith para el monitoreo y trazabilidad de las interacciones.

ESTRUCTURA DEL PROYECTO

chatbot-rag/  
├── src/  
│   ├── app.py              # Código principal del chatbot  
│   ├── rag.py              # Procesamiento de PDF y generación de embeddings  
│   └── memoria.py          # Gestión de memoria en sesiones  
├── data/  
│   └── documento.pdf       # PDF de ejemplo  
├── .gitignore              # Archivos excluidos del repositorio  
├── pyproject.toml          # Dependencias y configuración del proyecto  
├── .python-version         # Versión de Python 
├── README.md               # Documentación del proyecto  


REQUISITOS DEL SISTEMA 
- Python >= 3.11
- UV
- Ollama local
- Cuenta Langsmith
- Manejo de archivos Makefile

USO
Ejecutar make app 

- Abrir el Browser para ver la interfaz Streamlit
- Cargar PDF pero máximo 2000 MB
- Realizar una pregunta sobre el PDF cargado 


EJEMPLO DE CONSULTAS

PREGUNTA
Cuales son las consideraciones para la medicion a alta frecuencia ?
RESPUESTA
- Cada tipo de acelerónetro tiene sus propias frecuencias resonantes.
- Como las señales son muy pequeñas y de muy alta frecuencia entonces no se deben medir con las sondas.

PREGUNTA
Como es el procesamiento de la señal ?
RESPUESTA 
 El procesamiento de la señal de muy alta frecuencia varia segun cada fabricante, pero todos siguen el siguiente proceso:
 - Las señales se captan con un acelerómetro
 - Las señales que vienen del acelerometro son procesadas por un circuito especial que posee un filtro de paso de alta frecuencia.
