from fastapi import FastAPI, HTTPException
import torch
from pydantic import BaseModel, Field
import uvicorn

from const import MODEL_PATH, INPUT_SIZE, NUM_CLASSES
from model import SimpleNN

def load_model(model_path, input_size, num_classes=3):
    model = SimpleNN(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


model = load_model(MODEL_PATH, INPUT_SIZE, NUM_CLASSES)

app = FastAPI()

class InputData(BaseModel):
    features: list = Field(
        example=[-1.1245, 0.0000, 0.2033, 0.0000, -1.7930, 1.5171, -1.8123, 0.6345,
                 1.2402, 0.6318, -0.5228, -0.0738, -0.2208, -0.1441, -0.0398, -0.2293,
                 0.6193, 0.2154, -0.3252, -0.0251, -0.1970]
    )

@app.post("/predict")
async def predict(input_data: InputData):
    try:
        input_tensor = torch.tensor(input_data.features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0]
        return {"predicted_class": predicted_class, "confidence": confidence[predicted_class].item()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "predict API is running, send POST request to /predict endpoint with input data"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("FastAPI server is running")