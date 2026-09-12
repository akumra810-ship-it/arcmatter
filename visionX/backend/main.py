from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, shutil, json
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
os.makedirs("data/videos", exist_ok=True)
os.makedirs("data/frames", exist_ok=True)
app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")

# C++ Engine
engine = None
try:
    import visionx_engine
    engine = visionx_engine.AsyncVideoEngine()
    engine.load_model("yolov8n.onnx")
    print("C++ ENGINE LOADED WITH TRACKING + INTERP")
except Exception as e:
    print(f"Build C++ first: {e}")

class InterpReq(BaseModel):
    start_box: dict
    end_box: dict
    start_frame: int
    end_frame: int

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    path = f"data/videos/{file.filename}"
    with open(path, "wb") as out: shutil.copyfileobj(file.file, out)
    job_id = engine.submit_video(path, sample_fps=5)
    return {"job_id": job_id, "video_url": f"/videos/{file.filename}"}

@app.get("/video-job/{job_id}")
def job_status(job_id: str): return engine.get_video_status(job_id)

@app.get("/video-job/{job_id}/results")
def job_results(job_id: str):
    res = engine.get_video_results(job_id)
    grouped = []
    for frame_dets in res:
        grouped.append([{"x": d.x, "y": d.y, "w": d.w, "h": d.h, "class_id": d.class_id, "class_name": d.class_name, "frame_idx": d.frame_idx, "track_id": d.track_id, "conf": d.conf} for d in frame_dets])
    return {"grouped": grouped}

@app.post("/interpolate")
def interpolate(req: InterpReq):
    def dict_to_det(d, frame_idx):
        det = visionx_engine.Detection()
        det.x = int(d.get('x',0)); det.y = int(d.get('y',0)); det.w = int(d.get('width', d.get('w',0))); det.h = int(d.get('height', d.get('h',0)))
        det.class_id = int(d.get('class_id',0)); det.class_name = d.get('label', d.get('class_name','object')); det.track_id = int(d.get('track_id',1)); det.frame_idx = frame_idx
        return det
    start = dict_to_det(req.start_box, req.start_frame)
    end = dict_to_det(req.end_box, req.end_frame)
    result = engine.interpolate_keyframes(start, end, req.start_frame, req.end_frame)
    return [{"x": d.x, "y": d.y, "w": d.w, "h": d.h, "track_id": d.track_id, "frame_idx": d.frame_idx, "class_name": d.class_name} for d in result]

# Dummy endpoints for Annotator.jsx compatibility
@app.get("/frames/{video_id}/{frame_name}")
def get_frame(video_id: str, frame_name: str): return {"msg": "serve frame images if you extract them"}
@app.get("/annotations/{video_id}/{frame_name}")
def get_ann(video_id: str, frame_name: str): return []
@app.post("/annotations/{video_id}/{frame_name}")
def save_ann(video_id: str, frame_name: str, boxes: list): return {"saved": len(boxes)}