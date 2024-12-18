import argparse
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
import requests

from get_embedding_function import get_embedding_function

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Verify Gemini API key is set
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )
    
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    response = query_rag(query_text)
    print(response)

def query_rag(query_text: str):
    """
    Query the RAG system and return the response
    """
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize Gemini API call
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    # Updated data structure to match Gemini API requirements
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    # Updated headers (removed Bearer token)
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        raise Exception(f"Error from Gemini API: {response.text}")

    # Extract the response from the Gemini API response structure
    try:
        response_data = response.json()
        if 'candidates' in response_data:
            response_text = response_data['candidates'][0]['content']['parts'][0]['text']
        else:
            response_text = "No response generated"
    except Exception as e:
        raise Exception(f"Error parsing Gemini API response: {str(e)}")

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    return {
        "text": response_text,
        "sources": sources
    }

if __name__ == "__main__":
    main()
