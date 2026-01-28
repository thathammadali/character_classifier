from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import io
from PIL import Image
import model  # Import the model module we modified

app = FastAPI()

# --- CORS ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    explanation: str
    debug_image: str
    details: dict

@app.get("/")
def read_root():
    return {"message": "Character Classification API is running."}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Helper to convert label index to character (0-25 -> A-Z)
        # Check dataset mapping. A_Z usually maps 0=A...
        def get_char_label(idx):
            return chr(ord('A') + idx)

        # Call the analysis function
        # We access the globals from the 'model' module
        result = model.analyze_new_sample(
            contents,
            model.model,
            model.embedding_model,
            model.pca,
            model.kmeans,
            model.entropy_threshold,
            model.style_emnist,
            model.pred_emnist,
            model.entropy_emnist,
            model.y_test_emnist
        )
        
        # Construct Explanation
        pred_char = get_char_label(result['pred_label'])
        confidence_pct = result['confidence'] * 100
        
        explanation = f"I am {confidence_pct:.1f}% confident this is '{pred_char}'. "
        
        # Entropy check
        if result['rejected']:
            explanation += "However, the prediction is uncertain (high entropy), suggesting this style is ambiguous or unlike my training data. "
        else:
            explanation += "The prediction is stable (low entropy). "
            
        # Historical context
        hist_acc = result['historical_cluster_acc']
        if hist_acc >= 0:
            explanation += f"It matches a writing style cluster where I have a historical accuracy of {hist_acc*100:.1f}%."
        else:
            explanation += "It matches a common writing style cluster."

        return ClassificationResponse(
            label=pred_char,
            confidence=result['confidence'],
            explanation=explanation,
            debug_image=result.get('debug_image', ''),
            details=result
        )

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
