# Tree Health Classification (FastAPI + PyTorch + MLflow)

## Project Description
This project represents a RESTful API built using FastAPI for classifying the health status of trees (Good/Fair/Poor) based on data from the NY 2015 Street Tree Census. The model was trained using PyTorch, and experiments and hyperparameters were tracked using MLflow.
The project includes three main components:
1. A neural network architecture `SimpleNN` was built with **CrossEntropyLoss** and the **SGD optimizer**.
2. All training parameters and metrics are logged using **MLflow**.
3. Implemented via **FastAPI** for making predictions on new data.
---
## Instructions for Running
### Requirements
Before starting, ensure that you have the following packages installed:
- Python 3.10+ (It is recommended to use a virtual environment such as venv/Conda)
- CUDA (if using GPU for training)
- Dependencies listed in `requirements.txt`
Install the necessary packages using the command:
```bash
pip install -r requirements.txt
```
### Running Model Training
1. Prepared data is located in the `data/` folder. The file `ready.csv` is a reduced and preprocessed version of the dataset from the original Kaggle repository.
2. The model training script is located in the file `training.ipynb`.
3. After training is complete, the model will be saved in the file `model.pth`.
### Running the API
1. Ensure that the model file (`model.pth`) is available.
2. Start the API:
   ```bash
   python app.py
   ```
3. The API will be available at: `http://127.0.0.1:8000`.
---
## Example of API Usage
### Endpoints
#### 1. `/predict`
**Method:** POST  
**Description:** Performs prediction of tree health status based on input data.  
**Example Request:**
```json
{
  "features": [-1.1245, 0.0000, 0.2033, 0.0000, -1.7930, 1.5171, -1.8123, 0.6345,
               1.2402, 0.6318, -0.5228, -0.0738, -0.2208, -0.1441, -0.0398, -0.2293,
               0.6193, 0.2154, -0.3252, -0.0251, -0.1970]
}
```
**Example Response:**
```json
{
  "predicted_class": 1,
  "confidence": 0.85
}
```
#### 2. `/health`
**Method:** GET  
**Description:** Checks the API's operational status.  
**Example Response:**
```json
{
  "status": "healthy"
}
```
#### 3. `/`
**Method:** GET  
**Description:** Root endpoint providing information about the service.  
**Example Response:**
```json
{
  "message": "predict API is running, send POST request to /predict endpoint with input data"
}
```
---
## Model Architecture
The `SimpleNN` model is a simple multilayer perceptron (MLP) with three fully connected layers:
- **Input Layer:** Size equals `input_size`, determined by the number of features in the data.
- **Hidden Layers:**
  - First hidden layer: 128 neurons.
  - Second hidden layer: 64 neurons.
  - Activation: ReLU.
  - Dropout: Probability of 20% to prevent overfitting.
- **Output Layer:** Size equals `num_classes` (3 classes: Good, Fair, Poor).
- **Output Layer Activation:** Logits (softmax is applied during inference).
---
## Technical Details
### 1. Model Training
- **Framework:** PyTorch
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** SGD with momentum = 0.9
- **Hyperparameters:**
  - Learning rate: 0.00001
  - Weight decay: 1e-4
  - Batch size: 1024
  - Epochs: 100
  - Dropout rate: 0.2

**Note:**
All hyperparameters can be configured in the `config.yaml` file.

### 2. Experiment Logging
- **Tool:** MLflow
- **Logged Information:**
  - Model hyperparameters.
  - Training metrics (loss).
  - Saved artifacts (models).

### 3. API Deployment
- **Framework:** FastAPI
- **Server:** Uvicorn is used as the ASGI server.
---
## Examples of Usage
### Swagger UI
To test the API and perform requests, you can use Swagger UI, which is included as part of FastAPI. After starting the API, navigate to: `http://127.0.0.1:8000/docs`.
### cURL Example
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "features": [-1.1245, 0.0000, 0.2033, 0.0000, -1.7930, 1.5171, -1.8123, 0.6345,
               1.2402, 0.6318, -0.5228, -0.0738, -0.2208, -0.1441, -0.0398, -0.2293,
               0.6193, 0.2154, -0.3252, -0.0251, -0.1970]
}'
```

### 4. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```