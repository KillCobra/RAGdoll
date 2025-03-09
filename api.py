from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

@app.post("/process-scenario")
async def process_scenario(request: AnalysisRequest):
    try:
        # Process the data using your existing logic
        from query_data import run_query_with_description
        response = run_query_with_description(
            request.description,
            request.market_or_sector
        )

        return JSONResponse(content={
            "text": response["text"],
            "sources": response.get("sources", [])
        }, status_code=200)

    except Exception as e:
        print(f"Error processing scenario: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )