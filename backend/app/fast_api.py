from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from .agent import get_response # This import is correct

# --- API Data Structure ---
# --- THIS IS THE FIX ---
# We no longer require 'model_name' or 'model_provider' from the frontend.
class RequestState(BaseModel):
    system_prompt: str
    messages: List[str]
    allow_search: bool
    image_data: Optional[str] = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Team AI API",
    description="A robust multi-agent LangGraph API with image analysis and Tavily search.",
    version="6.0" # Version bump for dynamic model routing
)

# --- Health Check ---
@app.get("/")
def root():
    return {"status": "ok"}

# --- Inference Endpoint ---
@app.post("/agent")
def agent_endpoint(req: RequestState):
    """
    This endpoint now passes the request directly to the agent graph,
    which will handle model selection internally.
    """
    try:
        # --- THIS IS THE FIX ---
        # We no longer pass 'model_name' or 'model_provider'
        response_content = get_response(
            system_prompt=req.system_prompt,
            messages=req.messages,
            allow_search=req.allow_search,
            image_data=req.image_data 
        )
        return {"response": response_content}
    except Exception as e:
        print(f"An error occurred in the agent: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)