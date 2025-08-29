# Flower Classifier 🌸

Small project showing MLOps skills : API FastAPI + HTML page to **predict** flower type based on quantitative characteristics (open dataset Iris).

## 0) Setting up the virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Training the model
```bash
python train_model.py
```

## 2) Launching the API 
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Quick tests :
```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

## 3) If local use : 
- Open `public/index.html` with Live Server (has to be on 5500)
- Change `API_URL` if the API is on another machine.

## 4) Docker (don't forget the port mapping)
```bash
docker build -t iris-api .
docker run -e FLOWER_API_KEY=secret_key -p 8000:8000 iris-api
```

## 5) Deployment
- As of 2025-08-29, the API is hosted on Render : https://flower-api-daaw.onrender.com

---
 Thanks for reading through !