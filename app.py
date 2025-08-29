from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib, json, numpy as np, os

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from starlette.middleware.base import BaseHTTPMiddleware

API_KEY = os.getenv("IRIS_API_KEY")

# Loading model and its metadata
with open("model_meta.json", "r", encoding="utf-8") as f:
    META = json.load(f)
FEATURE_ORDER = META["feature_order"]
CLASS_NAMES = META["classes"]

MODEL = joblib.load("model.joblib")

class FlowerDims(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20)
    sepal_width:  float = Field(..., gt=0, lt=20)
    petal_length: float = Field(..., gt=0, lt=20)
    petal_width:  float = Field(..., gt=0, lt=20)

app = FastAPI(title="Iris API", version=META.get("model_version", "1.0"))


ALLOWED_ORIGINS = [
    "https://ian-geee.github.io",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type", "x-api-key"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class BodySizeLimiter(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int = 8 * 1024):
        super().__init__(app)
        self.max = max_bytes

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > self.max:
            return JSONResponse({"detail": "Payload too large"}, status_code=413)
        body = await request.body()
        if body and len(body) > self.max:
            return JSONResponse({"detail": "Payload too large"}, status_code=413)
        return await call_next(request)

app.add_middleware(BodySizeLimiter)

def check_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/")
def root():
    return {"ok": True, "message": "Use GET /health or POST /predict"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": META.get("model_type"),
        "model_version": META.get("model_version"),
        "classes": CLASS_NAMES if CLASS_NAMES else [int(c) for c in MODEL.classes_],
    }

@limiter.limit("30/minute")
@app.post("/predict")
def predict(
    payload: FlowerDims,
    request: Request,
    x_api_key: str | None = Header(default=None, convert_underscores=False),
):
    check_api_key(x_api_key)

    x = np.array([[getattr(payload, f) for f in FEATURE_ORDER]])
    try:
        proba = MODEL.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    return {
        "predicted_class_index": pred_idx,
        "predicted_class_label": CLASS_NAMES[pred_idx],
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
        "feature_order": FEATURE_ORDER
    }
