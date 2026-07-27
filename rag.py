"""Logica central del agente RAG sobre el Plan de Estudios (PDF).

Este modulo concentra la configuracion y las funciones reutilizables tanto por
el script de ingesta (ingest.py) como por la interfaz web (app.py).
"""

from __future__ import annotations

import os

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Configuracion general ---------------------------------------------------
PDF_PATH = "plan_estudios_maestria_ciencias_2023.pdf"
INDEX_DIR = "data/faiss_index"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4  # cuantos fragmentos recuperar por pregunta

SYSTEM_PROMPT = (
    "Eres un asistente experto que responde preguntas sobre el Plan de Estudios "
    "de la Maestria en Ciencias de la Universidad Autonoma del Estado de Morelos "
    "(UAEM), 2023.\n\n"
    "Reglas:\n"
    "- Responde SIEMPRE en espanol, de forma clara y bien estructurada.\n"
    "- Basate UNICAMENTE en el contexto proporcionado del documento.\n"
    "- Si la respuesta no esta en el contexto, dilo con honestidad: 'No encontre "
    "esa informacion en el documento.' No inventes datos.\n"
    "- Cuando sea util, menciona la pagina del documento de donde sale la "
    "informacion.\n\n"
    "Contexto del documento:\n{context}"
)


def get_embeddings() -> OpenAIEmbeddings:
    """Modelo de embeddings de OpenAI."""
    return OpenAIEmbeddings(model=EMBED_MODEL)


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Modelo de chat de OpenAI."""
    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature)


def load_vectorstore() -> FAISS:
    """Carga el indice FAISS ya construido desde disco."""
    if not os.path.isdir(INDEX_DIR):
        raise FileNotFoundError(
            f"No existe el indice en '{INDEX_DIR}'. Ejecuta primero: python ingest.py"
        )
    return FAISS.load_local(
        INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def build_qa_chain(vectorstore: FAISS):
    """Construye la cadena de pregunta-respuesta con recuperacion de contexto."""
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(get_llm(), prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


def answer_question(chain, question: str) -> dict:
    """Ejecuta una pregunta y devuelve la respuesta y los documentos fuente."""
    result = chain.invoke({"input": question})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("context", []),
    }
