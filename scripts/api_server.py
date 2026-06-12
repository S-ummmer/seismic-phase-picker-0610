# scripts/api_server.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\api_server.py

"""
地震震相拾取 HTTP API 服务。

启动:
    python scripts/api_server.py --config config.yaml --port 8000

端点:
    POST /predict    上传波形 HDF5，返回拾取结果
    GET  /health     健康检查
"""

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import h5py
import numpy as np

from src.pipeline import SeismicPipeline
from src.io import Waveform


# --- 全局变量 ---
pipeline: SeismicPipeline = None

app = FastAPI(title="Seismic Phase Picker API", version="1.0.0")


class PickedPhaseResponse(BaseModel):
    time: float
    phase: str
    probability: float


class PredictionResponse(BaseModel):
    event_id: str
    n_picks: int
    picks: List[PickedPhaseResponse]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """上传 HDF5 波形文件，返回震相拾取结果。"""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        with h5py.File(tmp_path, "r") as f:
            data = f["data"][:]
            sr = f["data"].attrs.get("sampling_rate", 100.0)
            start_time = f["data"].attrs.get("start_time", 0.0)

        wf = Waveform(data=data, sampling_rate=sr, start_time=start_time)
        picks = pipeline.run_inference(wf)

        return PredictionResponse(
            event_id=file.filename.rsplit(".", 1)[0],
            n_picks=len(picks),
            picks=[
                PickedPhaseResponse(time=p.time, phase=p.phase, probability=p.probability)
                for p in picks
            ],
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main():
    global pipeline
    parser = argparse.ArgumentParser(description="Seismic Phase Picker API Server")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    pipeline = SeismicPipeline(args.config)
    print(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
