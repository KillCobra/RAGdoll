# RAG Tutorial V2

This project is a scenario-based RAG pipeline that generates a risk analysis report for a business expansion idea.

The main flow is:

1. Take a company description and a target market or sector.
2. Extract keywords from the description.
3. Search Google Scholar for related PDF sources.
4. Load PDFs from the local `data/` folder into Chroma.
5. Retrieve relevant context from Chroma.
6. Ask Gemini to generate a risk-focused report with mitigation ideas.

The `data/` folder is the local document corpus for the vector store. It is used when building the Chroma database, so the project can answer questions grounded in those documents.

## Can I ask my own questions over the data in `data/`?

Yes. The project now supports a separate CLI mode for one-file RAG chat. That mode lets you pick a PDF from `data/`, loads it into Chroma, and then lets you ask multiple follow-up questions with quoted line references from the selected file.

The default command in `main.py` still does the scenario-based risk report from a company description plus a target market or sector. The frontend and FastAPI route also call that same scenario-analysis path.

## Prerequisites

- Python 3.10+ recommended.
- A valid `GEMINI_API_KEY` in a `.env` file at the repository root.
- Node.js 18+ for the frontend.

Example `.env`:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_API_VERSION=v1beta
GEMINI_BASE_URL=https://generativelanguage.googleapis.com
GEMINI_MAX_RETRIES=5
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

## Core Commands

### Run the main analysis CLI

```bash
python main.py "We are a sustainable energy company focusing on solar panel manufacturing and renewable energy solutions for residential buildings." "Residential Area"
```

Optional flags:

```bash
python main.py "Company description" "Target sector" --max_keywords 6 --max_pdf_links 20
```

This is the main supported entry point. It builds search queries from the company description, scrapes related PDFs, adds them to Chroma, and then generates a report.

The Gemini model is configurable through `GEMINI_MODEL` in `.env`, so you can switch away from `gemini-1.5-flash-latest` if your account does not support that model name.

### Populate the local vector database from `data/`

```bash
python populate_database.py
```

Useful options:

```bash
python populate_database.py --reset
python populate_database.py --view
```

- `--reset` clears the existing `chroma/` database.
- `--view` exports the current Chroma contents to `data/documents.txt`.

### Start the FastAPI backend

```bash
uvicorn api:app --reload
```

This exposes `POST /process-scenario`, which runs the same analysis flow as `main.py`.

### Start the Next.js frontend

Run this from the `frontend/` folder:

```bash
npm run dev
```

The frontend calls the backend on `http://localhost:8000/process-scenario`.

### Test the Google Scholar PDF scraper

```bash
python scrapper.py
```

This is an interactive utility that prints PDF links for a search term.

### Run the local single-PDF chat mode

```bash
python local_chat.py
```

Optional tuning flags:

```bash
python local_chat.py --top_k 4 --history_turns 4
```

This mode prompts you to choose a PDF from `data/`, indexes that file into a local Chroma store, and then starts a multi-turn chat that answers from the file itself with quoted line references.

## Other Scripts

- `query_data.py` contains the lower-level Chroma + Gemini query helpers used by the analysis flow. Its standalone CLI is not currently a polished entry point.
- `local_chat.py` is the CLI chat mode for asking questions about one selected PDF from `data/`.
- `test_scraper.py` is a prototype interactive Q&A loop over Chroma and Gemini.
- `get_embedding_function.py` provides the embedding function used by Chroma.

## Project Structure

- `main.py` - orchestrates keyword extraction, PDF scraping, ingestion, and report generation.
- `query_data.py` - performs retrieval and Gemini prompting.
- `populate_database.py` - loads PDF files from `data/` into Chroma.
- `scrapper.py` - Google Scholar PDF scraper.
- `api.py` - FastAPI wrapper for the analysis pipeline.
- `frontend/` - Next.js UI that calls the backend API.
- `data/` - local PDFs used to seed the vector database.
- `chroma/` - persisted Chroma database.

## Configuration

The main runtime settings live in `.env`:

- `GEMINI_API_KEY` - required API key.
- `GEMINI_MODEL` - Gemini model name used by the analysis and chat flows.
- `GEMINI_API_VERSION` - Gemini API version path, default `v1beta`.
- `GEMINI_BASE_URL` - Gemini API base URL.
- `GEMINI_MAX_RETRIES` - retry count for Gemini requests.
- `EMBEDDING_MODEL_NAME` - Hugging Face embedding model used for Chroma.

## How the Answering Works

The current code uses Chroma for retrieval and Gemini for generation. The answer is conditioned on:

- the company description you provide,
- the target market or sector,
- and the retrieved context from the indexed documents.

So the system is best described as a risk-advisory RAG app plus a separate single-PDF document chat mode. The analysis flow still expects a business scenario and produces a report around expansion risks and mitigations.