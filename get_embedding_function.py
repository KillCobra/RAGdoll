from langchain_huggingface import HuggingFaceEmbeddings
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_embedding_function():
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings
