from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
import sys
import pandas as pd
from pydantic import BaseModel
import requests

from app.preprocessing import bool_cols, transform_features

sys.modules["__main__"].transform_features = transform_features

app = FastAPI(title="UK Road Safety - Collision Severity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "accident_severity_pipeline.joblib"
MODEL_URL = "https://github.com/AmiraTarek815/Depi-Final-Project/releases/download/v1.0/accident_severity_pipeline.joblib"


def ensure_model_downloaded(dest_path: Path, url: str):
  if dest_path.exists() and dest_path.stat().st_size > 0:
    return
  dest_path.parent.mkdir(parents=True, exist_ok=True)
  response = requests.get(url, stream=True, timeout=180)
  response.raise_for_status()
  with open(dest_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
      if chunk:
        f.write(chunk)


ensure_model_downloaded(MODEL_PATH, MODEL_URL)

pipeline = joblib.load(MODEL_PATH)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

SEVERITY_LABELS = {0: "Slight", 1: "Serious", 2: "Fatal"}


class AccidentInput(BaseModel):
  longitude: float
  latitude: float
  number_of_vehicles: int
  first_road_class: str
  road_type: str
  speed_limit: int
  junction_detail: str
  second_road_class: str
  pedestrian_crossing: str
  light_conditions: str
  weather_conditions: str
  road_surface_conditions: str
  urban_or_rural_area: str
  did_police_officer_attend_scene_of_accident: str
  trunk_road_flag: str
  hour: int
  month: int
  is_weekend: int
  has_two_wheeler: bool
  any_overturned: bool
  any_left_road: bool
  max_engine_cc: float
  has_pedestrian: bool
  mean_casualty_age: float


@app.get("/")
def health_check():
  return {"status": "running", "message": "Collision Severity API is ready"}


@app.post("/predict")
def predict_severity(data: AccidentInput):
  input_df = pd.DataFrame([data.model_dump()])
  pred_idx = int(pipeline.predict(input_df)[0])
  probs = pipeline.predict_proba(input_df)[0]

  return {
      "severity_code": pred_idx,
      "severity_label": SEVERITY_LABELS[pred_idx],
      "probabilities": {
          "Slight": round(float(probs[0]) * 100, 2),
          "Serious": round(float(probs[1]) * 100, 2),
          "Fatal": round(float(probs[2]) * 100, 2),
      },
  }