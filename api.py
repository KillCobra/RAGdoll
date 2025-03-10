from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from main import run_analysis  # Import the run_analysis function

app = FastAPI()

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class AnalysisRequest(BaseModel):
    description: str
    market_or_sector: str
    max_keywords: int = 6
    max_pdf_links: int = 20

@app.post("/process-scenario")
async def process_scenario(request: AnalysisRequest):
    try:
        # Call the run_analysis function and await its result
        response = await asyncio.to_thread(run_analysis, 
            request.description,
            request.market_or_sector,
            request.max_keywords,
            request.max_pdf_links
        )

        return JSONResponse(content={
            "text": response["text"],
            "sources": response.get("sources", [])
        }, status_code=200)

    except Exception as e:
        print(f"Error processing scenario: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)