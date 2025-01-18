import argparse
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
import requests
import time

from get_embedding_function import get_embedding_function
from langchain.schema.document import Document

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Updated prompt template to include company description
PROMPT_TEMPLATE = """
Company Description:
{description}
Context:
{context}
---

Question:
{question}

Based on the company description and the context provided, please answer the question.
"""

def run_query_with_description(company_description, market_or_sector, query_text=None):
    if query_text is None:
        query_text = (f"Could you provide an analysis of the key risks involved in expanding into {market_or_sector}? "
                       "Outline the potential risks and their impacts on the business. Also, analyze potential risk "
                       "mitigation strategies and prioritize them based on effectiveness and feasibility. "
                       "Please list them in order of importance and provide any recommendations for mitigating the most critical risks.")
    return main(company_description, query_text)

def main(company_description, query_text):
    # Verify Gemini API key is set
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )
    
    # Create CLI.
    parser = argparse.ArgumentParser(description="Query data based on company description.")
    parser.add_argument("company_description", type=str, nargs='?', default=company_description, help="Company description")
    parser.add_argument("market_or_sector", type=str, nargs='?', default="", help="Market or sector for expansion")
    args = parser.parse_args()

    response = query_rag(args.company_description, query_text)
    if response:
        print("\nResponse:", response["text"])
        print("\nSources:", response["sources"])
    else:
        print("No response generated.")

def query_rag(company_description: str, query_text: str):
    """
    Query the RAG system and return the response
    """
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # # Check if the query_text has already been used
    # if is_keyword_used(db, query_text):
    #     print(f"Query '{query_text}' has already been used. Skipping search.")
    #     return {"text": "No new search performed.", "sources": []}
        

    # Proceed with the similarity search
    results = db.similarity_search_with_score(company_description, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text, description=company_description)

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
    
    # Retry logic with exponential backoff
    for attempt in range(5):  # Try up to 5 times
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            break  # Exit the loop if the request was successful
        elif response.status_code == 429:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        else:
            raise Exception(f"Error from Gemini API: {response.text}")
    else:
        raise Exception("Failed to retrieve content after multiple attempts due to rate limiting.")

    # Extract the response from the Gemini API response structure
    try:
        response_data = response.json()
        response_text = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response generated")
    except Exception as e:
        raise Exception(f"Error parsing Gemini API response: {str(e)}")

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    # Add the query_text as a used keyword to Chroma DB
    add_keyword(db, query_text)

    return {
        "text": response_text,
        "sources": sources
    }

def is_keyword_used(db, keyword, threshold=0.8):
    """
    Check if the keyword has already been used by searching in Chroma DB.
    """
    results = db.similarity_search_with_score(keyword, k=1)  # We check for the most similar document
    if results:
        top_doc, score = results[0]
        if score > threshold:  # If the match is good enough (score threshold can be adjusted)
            print(f"Keyword '{keyword}' found in Chroma DB with a score of {score}. Skipping search.")
            return True
    return False


def add_keyword(db, keyword):
    """
    Add the keyword to Chroma DB to mark it as used.
    """
    keyword_doc = Document(
        page_content=keyword,
        metadata={"type": "keyword"}
    )
    db.add_documents([keyword_doc])
    print(f"Keyword '{keyword}' added to Chroma DB as used.")

if __name__ == "__main__":
    run_query_with_description(
        "We are a sustainable energy company focusing on solar panel manufacturing and renewable energy solutions for residential buildings.",
        "Residential Area",
        query_text=None
    )