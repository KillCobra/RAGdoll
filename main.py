import fitz  # PyMuPDF
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Extract text from the PDF
pdf_text = extract_text_from_pdf("C:/Users/pranjalpratosh/Downloads/rag-tutorial-v2/data/2-Module 1-01-08-2024.pdf")

# Prepare the request to the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=" + api_key

# Prepare the data for the API request
data = {
    "prompt": pdf_text,
    "max_tokens": 512,
    "temperature": 0.5
}

# Make the API request
response = requests.post(url, json=data)

# Check for errors in the response
if response.status_code != 200:
    raise Exception(f"Error from Gemini API: {response.text}")

# Extract the generated audio content from the response
audio_content = response.json().get("audio", None)

# Save the output if audio content is available
if audio_content:
    with open("output.wav", "wb") as audio_file:
        audio_file.write(audio_content)
else:
    print("No audio content returned from the API.")