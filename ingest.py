"""Construye el indice vectorial (FAISS) a partir del PDF del plan de estudios.

Uso:
    python ingest.py

Lee el PDF, lo divide en fragmentos, genera los embeddings con OpenAI y guarda
el indice en disco (carpeta data/faiss_index) para que la app lo cargue sin
tener que reprocesar el documento cada vez.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import CHUNK_OVERLAP, CHUNK_SIZE, INDEX_DIR, PDF_PATH, get_embeddings


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Falta OPENAI_API_KEY. Crea un archivo .env con tu clave "
            "(mira .env.example)."
        )

    if not os.path.exists(PDF_PATH):
        raise SystemExit(f"No encontre el PDF: {PDF_PATH}")

    print(f"Leyendo PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"Paginas cargadas: {len(pages)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"Fragmentos generados: {len(chunks)}")

    print("Generando embeddings y construyendo el indice FAISS (puede tardar)...")
    start = time.time()
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    print(f"Indice construido en {time.time() - start:.1f} s")

    os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"Indice guardado en: {INDEX_DIR}")


if __name__ == "__main__":
    main()
