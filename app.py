import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from planner import generate_experiment_plan

app = FastAPI(title="The AI Scientist API")

# Enable CORS so Lovable can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for the generator
class HypothesisRequest(BaseModel):
    hypothesis: str

@app.post("/api/generate")
def api_generate_plan(request: HypothesisRequest):
    print(f"📥 Received request for hypothesis: {request.hypothesis}")
    # Triggers the search and planning pipeline
    result = generate_experiment_plan(request.hypothesis)
    return result

@app.post("/api/review")
async def save_review(request: Request):
    try:
        # Capture the raw JSON payload from Lovable
        data = await request.json()
        
        print("🚀 NEW SCIENTIST FEEDBACK RECEIVED!")
        print(json.dumps(data, indent=2))
        
        # Save feedback to a log file for the judges to see
        with open("scientist_feedback_log.json", "a") as f:
            f.write(json.dumps(data) + "\n")
            
        return {"status": "success", "message": "Feedback captured and saved."}
    except Exception as e:
        print(f"❌ Error saving review: {e}")
        return {"status": "error", "message": str(e)}

# Run this using: uvicorn app:app --reload