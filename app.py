"""Interfaz web (Streamlit) del agente RAG sobre el Plan de Estudios.

Ejecutar localmente:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from rag import answer_question, build_qa_chain, load_vectorstore

load_dotenv()

# En Streamlit Community Cloud la clave se define en "Secrets" (st.secrets).
# Localmente se toma del archivo .env.
if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Agente Plan de Estudios - Maestria en Ciencias", page_icon="🎓")


@st.cache_resource(show_spinner="Cargando el indice del documento...")
def get_chain():
    vectorstore = load_vectorstore()
    return build_qa_chain(vectorstore)


st.title("🎓 Agente RAG — Maestría en Ciencias (UAEM)")
st.caption(
    "Pregunta lo que quieras sobre el Plan de Estudios de la Maestría en Ciencias "
    "de la UAEM (2023). El agente responde basándose en el documento."
)

with st.sidebar:
    st.header("ℹ️ Acerca de")
    st.markdown(
        "- **Fuente:** Plan de Estudios – Maestría en Ciencias, UAEM (2023)\n"
        "- **Técnica:** RAG (Retrieval-Augmented Generation)\n"
        "- **Modelo:** OpenAI `gpt-4o-mini`\n"
        "- **Búsqueda:** FAISS + embeddings `text-embedding-3-small`"
    )
    st.divider()
    st.subheader("💡 Ejemplos de preguntas")
    st.markdown(
        "- ¿Cuánto dura la maestría y cuál es su modalidad?\n"
        "- ¿Cuáles son los requisitos de ingreso?\n"
        "- ¿Qué líneas de investigación (LGAC) ofrece el programa?\n"
        "- ¿Cuáles son los requisitos para obtener el grado?"
    )
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "No se encontró la variable OPENAI_API_KEY. Configúrala en el archivo .env "
        "(local) o en los Secrets de Streamlit Cloud."
    )
    st.stop()

try:
    chain = get_chain()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Escribe tu pregunta sobre el plan de estudios..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el documento y redactando la respuesta..."):
            result = answer_question(chain, question)
        st.markdown(result["answer"])

        sources = result.get("sources", [])
        if sources:
            with st.expander("📄 Fuentes del documento"):
                for doc in sources:
                    page = doc.metadata.get("page")
                    page_label = f"Página {page + 1}" if isinstance(page, int) else "Fragmento"
                    snippet = doc.page_content.strip().replace("\n", " ")[:300]
                    st.markdown(f"**{page_label}:** {snippet}…")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )
