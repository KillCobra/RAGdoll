import argparse
import hashlib
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain.schema.document import Document

from get_embedding_function import get_embedding_function
from llm_client import generate_gemini_text

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


DATA_PATH = Path("data")
LOCAL_CHAT_PATH = Path("chroma") / "local_chat"
DEFAULT_TOP_K = 4
DEFAULT_HISTORY_TURNS = 4
DEFAULT_LINES_PER_CHUNK = 12


def list_pdf_files() -> list[Path]:
    return sorted(DATA_PATH.rglob("*.pdf"))


def choose_pdf_file() -> Path:
    pdf_files = list_pdf_files()
    if not pdf_files:
        raise FileNotFoundError("No PDF files were found in the data directory.")

    print("Available PDF files:")
    for index, pdf_file in enumerate(pdf_files, start=1):
        print(f"  {index}. {pdf_file.relative_to(DATA_PATH)}")

    while True:
        selection = input("Select a file by number or relative path: ").strip()
        if not selection:
            continue

        if selection.isdigit():
            index = int(selection)
            if 1 <= index <= len(pdf_files):
                return pdf_files[index - 1]

        candidates = [Path(selection), DATA_PATH / Path(selection)]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".pdf":
                return candidate

        print("Invalid selection. Try again.")


def safe_collection_name(pdf_path: Path) -> str:
    digest = hashlib.sha1(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return digest


def chunk_page_lines(page_text: str, page_number: int, source_name: str, lines_per_chunk: int) -> list[Document]:
    lines = [line.strip() for line in page_text.splitlines()]
    lines = [line for line in lines if line]

    documents = []
    for start_index in range(0, len(lines), lines_per_chunk):
        chunk_lines = lines[start_index : start_index + lines_per_chunk]
        if not chunk_lines:
            continue

        line_start = start_index + 1
        line_end = start_index + len(chunk_lines)
        numbered_text = "\n".join(
            f"Line {line_start + offset}: {line}" for offset, line in enumerate(chunk_lines)
        )

        documents.append(
            Document(
                page_content=f"Source: {source_name}\nPage: {page_number}\n{numbered_text}",
                metadata={
                    "source": source_name,
                    "page": page_number,
                    "line_start": line_start,
                    "line_end": line_end,
                },
            )
        )

    return documents


def load_pdf_documents(pdf_path: Path, lines_per_chunk: int = DEFAULT_LINES_PER_CHUNK) -> list[Document]:
    documents = []
    with fitz.open(str(pdf_path)) as pdf_document:
        for page_index in range(len(pdf_document)):
            page = pdf_document[page_index]
            page_text = page.get_text("text")
            documents.extend(
                chunk_page_lines(
                    page_text=page_text,
                    page_number=page_index + 1,
                    source_name=pdf_path.name,
                    lines_per_chunk=lines_per_chunk,
                )
            )

    return documents


def ensure_pdf_index(pdf_path: Path) -> Chroma:
    embedding_function = get_embedding_function()
    persist_directory = LOCAL_CHAT_PATH / safe_collection_name(pdf_path)
    db = Chroma(persist_directory=str(persist_directory), embedding_function=embedding_function)

    existing_items = db.get(include=[])
    if existing_items.get("ids"):
        return db

    documents = load_pdf_documents(pdf_path)
    if not documents:
        raise ValueError(f"No text could be extracted from {pdf_path.name}.")

    ids = []
    for document in documents:
        ids.append(
            f"{pdf_path.name}:p{document.metadata['page']}:l{document.metadata['line_start']}-{document.metadata['line_end']}"
        )

    db.add_documents(documents, ids=ids)
    return db


def format_context(results) -> str:
    blocks = []
    for document, score in results:
        metadata = document.metadata
        blocks.append(
            "\n".join(
                [
                    f"[score={score:.3f}] {metadata.get('source', 'unknown')} page {metadata.get('page', '?')} lines {metadata.get('line_start', '?')}-{metadata.get('line_end', '?')}",
                    document.page_content,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def format_history(history: list[dict[str, str]], max_turns: int) -> str:
    if not history:
        return "No prior turns."

    recent_turns = history[-max_turns:]
    formatted_turns = []
    for turn in recent_turns:
        formatted_turns.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
    return "\n\n".join(formatted_turns)


def build_prompt(history: str, context: str, question: str, file_name: str) -> str:
    return f"""You are a careful RAG assistant for a single PDF file.
Answer only from the document context and the conversation history.
If the context does not support an answer, say that you cannot verify it from {file_name}.

Rules:
- Quote the exact supporting lines from the document when possible.
- Include citations in the form [file | page X | lines Y-Z].
- Keep answers grounded in the provided text.
- Use the conversation history only to maintain context across turns, not as a source of facts.

Conversation history:
{history}

Document context:
{context}

Question:
{question}
"""


def run_local_chat(top_k: int = DEFAULT_TOP_K, history_turns: int = DEFAULT_HISTORY_TURNS) -> None:
    pdf_path = choose_pdf_file()
    print(f"Loading {pdf_path.relative_to(DATA_PATH)} into Chroma...")
    db = ensure_pdf_index(pdf_path)
    print("Ready. Ask questions about the selected file. Type 'exit' to quit or 'file' to choose another PDF.")

    history: list[dict[str, str]] = []

    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        if question.lower() == "file":
            pdf_path = choose_pdf_file()
            print(f"Loading {pdf_path.relative_to(DATA_PATH)} into Chroma...")
            db = ensure_pdf_index(pdf_path)
            history.clear()
            continue

        results = db.similarity_search_with_score(question, k=top_k)
        context = format_context(results)
        history_text = format_history(history, history_turns)
        prompt = build_prompt(history_text, context, question, pdf_path.name)
        answer = generate_gemini_text(prompt)

        print("\nAnswer:\n")
        print(answer)

        history.append({"question": question, "answer": answer})


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a single PDF from the data folder.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of retrieved chunks to include.")
    parser.add_argument(
        "--history_turns",
        type=int,
        default=DEFAULT_HISTORY_TURNS,
        help="Number of recent turns to keep in the prompt.",
    )
    args = parser.parse_args()
    run_local_chat(top_k=args.top_k, history_turns=args.history_turns)


if __name__ == "__main__":
    main()