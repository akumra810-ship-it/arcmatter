import { useState, useEffect } from "react";
import { Stage, Layer, Rect, Text } from "react-konva";

const API = "http://localhost:8000";

export default function Annotator({ videoId, frames }) {
  const [idx, setIdx] = useState(0);
  const [boxes, setBoxes] = useState([]);
  const [drawing, setDrawing] = useState(null);
  const [loading, setLoading] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [grouped, setGrouped] = useState({}); // {frameName: [boxes]}
  const [keyframes, setKeyframes] = useState({}); // {track_id: [{frame_idx, box}]}

  const frameName = frames[idx];

  useEffect(() => {
    if (!frameName) return;
    if (grouped[frameName]) {
      setBoxes(grouped[frameName]);
    } else {
      fetch(`${API}/annotations/${videoId}/${frameName}`)
        .then((r) => r.json())
        .then((data) => {
          const b = Array.isArray(data) ? data : [];
          setBoxes(b);
        });
    }
  }, [frameName]);

  const save = async () => {
    setGrouped((g) => ({ ...g, [frameName]: boxes }));
    await fetch(`${API}/annotations/${videoId}/${frameName}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(boxes),
    });
  };

  // --- Auto This Frame ---
  const autoLabel = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/auto-label/${videoId}/${frameName}`, { method: "POST" });
      const data = await res.json();
      setBoxes(data.map(d => ({ x: d.x, y: d.y, width: d.w, height: d.h, label: d.class_name, conf: d.conf, auto: true, track_id: d.track_id })));
    } catch { alert("Auto-label failed"); }
    setLoading(false);
  };

  // --- Auto ALL Frames (C++ Async Video Engine with Tracking) ---
  const autoLabelAll = async () => {
    if (!confirm(`Auto-label all ${frames.length} frames with C++ Async + Tracking?`)) return;
    setLoading(true);
    try {
      const res = await fetch(`${API}/upload-video`, { method: "POST" }); // your backend should have videoId mapping
      // If your backend needs video path, use submit_video job
      const uploadRes = await fetch(`${API}/video-job/${videoId}/start`, { method: "POST" }).catch(async () => {
        // Fallback: call /upload-video job that we built earlier
        const fd = new FormData();
        // We use a dummy job for now, backend will process data/videos/demo.mp4
        const r = await fetch(`${API}/video-job/demo/start`, { method: "POST" });
        return r;
      });

      // Real flow for our C++ engine:
      const r = await fetch(`${API}/video-job/${frames.length > 0 ? 'job' : 'test'}/status`);
      // For final engine we built, use /upload-video endpoint which returns job_id
      // So we poll:
      const jobRes = await fetch(`${API}/video-job/${jobId || 'last'}`);
      // Simplified polling for our engine:
      const pollJob = async (id) => {
        const rr = await fetch(`${API}/video-job/${id}`);
        const s = await rr.json();
        if (s.error) return;
        setProgress(s.progress);
        setJobId(s.job_id);
        if (!s.done) {
          setTimeout(() => pollJob(id), 1000);
        } else {
          const resultRes = await fetch(`${API}/video-job/${id}/results`);
          const dd = await resultRes.json();
          const newGrouped = {};
          dd.grouped.forEach((frameDets, i) => {
            const fname = frames[i];
            if (!fname) return;
            newGrouped[fname] = frameDets.map(d => ({ x: d.x, y: d.y, width: d.w, height: d.h, label: d.class_name, conf: d.conf, track_id: d.track_id, auto: true }));
          });
          setGrouped(newGrouped);
          setBoxes(newGrouped[frameName] || []);
          setLoading(false);
        }
      };

      // Start a new video job - this is the actual endpoint from visionx_engine.cpp
      const startRes = await fetch(`${API}/upload-video`, { method: "POST", body: new FormData() }).catch(() => null);
      // For now use existing jobId if set
      if (jobId) pollJob(jobId);
      else {
        // Directly fetch results if you uploaded via App.jsx
        const res2 = await fetch(`${API}/video-job/${jobId}/results`);
        if (res2.ok) {
          const dd = await res2.json();
          console.log(dd);
        }
      }

    } catch (e) {
      console.error(e);
      alert("Bulk failed - use /upload-video flow from App.jsx");
      setLoading(false);
    }
  };

  // --- NEW: Interpolate between two keyframes ---
  const interpolate = async () => {
    if (boxes.length === 0) { alert("Draw a box first and make it a keyframe"); return; }
    const trackId = boxes[0].track_id || 1;
    const kfs = keyframes[trackId];
    if (!kfs || kfs.length < 1) {
      // Save current as first keyframe
      setKeyframes({ ...keyframes, [trackId]: [{ frame_idx: idx, box: boxes[0] }] });
      alert(`Saved keyframe at frame ${idx} for track ${trackId}. Now go to another frame, move box, and click Interpolate again.`);
      return;
    }
    const startKf = kfs[kfs.length - 1];
    const endKf = { frame_idx: idx, box: boxes[0] };

    try {
      const res = await fetch(`${API}/interpolate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_box: { ...startKf.box, track_id: trackId },
          end_box: { ...endKf.box, track_id: trackId },
          start_frame: startKf.frame_idx,
          end_frame: endKf.frame_idx,
        }),
      });
      const interpolated = await res.json(); // 31 boxes
      // Fill grouped cache
      const newGrouped = { ...grouped };
      interpolated.forEach(b => {
        const fIdx = b.frame_idx;
        const fname = frames[fIdx];
        if (!fname) return;
        newGrouped[fname] = [{ x: b.x, y: b.y, width: b.w, height: b.h, label: b.class_name, track_id: b.track_id, interpolated: true }];
      });
      setGrouped(newGrouped);
      alert(`Interpolated ${interpolated.length} frames for track ${trackId}`);
      setKeyframes({ ...keyframes, [trackId]: [...kfs, endKf] });
    } catch (e) {
      alert("Interpolation failed - is C++ engine running?");
    }
  };

  const trackToNext = async () => {
    if (idx >= frames.length - 1 || boxes.length === 0) return;
    setLoading(true);
    const nextFrame = frames[idx + 1];
    try {
      const res = await fetch(`${API}/track/${videoId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from_frame: frameName, to_frame: nextFrame, boxes }),
      });
      const tracked = await res.json();
      setIdx(idx + 1);
      const mapped = tracked.map(b => ({ ...b, tracked: true }));
      setBoxes(mapped);
      setGrouped(g => ({ ...g, [nextFrame]: mapped }));
    } catch { alert("Tracking failed"); }
    setLoading(false);
  };

  const exportCOCO = async () => {
    const allBoxes = Object.entries(grouped).flatMap(([fname, bxs]) => bxs.map(b => ({ ...b, frame: fname })));
    const blob = new Blob([JSON.stringify({ videoId, frames: Object.keys(grouped).length, boxes: allBoxes }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `${videoId}_coco.json`; a.click();
    URL.revokeObjectURL(url);
  };

  const handleMouseDown = (e) => {
    const pos = e.target.getStage().getPointerPosition();
    setDrawing({ x: pos.x, y: pos.y, width: 0, height: 0, label: "object", track_id: Date.now() % 10000 });
  };
  const handleMouseMove = (e) => {
    if (!drawing) return;
    const pos = e.target.getStage().getPointerPosition();
    setDrawing({ ...drawing, width: pos.x - drawing.x, height: pos.y - drawing.y });
  };
  const handleMouseUp = () => {
    if (drawing && Math.abs(drawing.width) > 10 && Math.abs(drawing.height) > 10) {
      const norm = {
        x: drawing.width < 0 ? drawing.x + drawing.width : drawing.x,
        y: drawing.height < 0 ? drawing.y + drawing.height : drawing.y,
        width: Math.abs(drawing.width),
        height: Math.abs(drawing.height),
        label: "object",
        track_id: drawing.track_id,
      };
      setBoxes([...boxes, norm]);
    }
    setDrawing(null);
  };

  return (
    <div style={{ background: "#0a0a0a", color: "#fff", minHeight: "100vh", padding: 12 }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap", alignItems: "center" }}>
        <button onClick={() => setIdx(Math.max(0, idx - 1))}>⬅️ Prev</button>
        <b>{idx + 1} / {frames.length}</b>
        <span style={{ opacity: 0.6, fontSize: 12 }}>{frameName}</span>
        <button onClick={() => setIdx(Math.min(frames.length - 1, idx + 1))}>Next ➡️</button>
        <button onClick={autoLabel} disabled={loading} style={{ background: "#6d28d9", color: "#fff" }}>{loading ? "..." : "🤖 Auto This"}</button>
        <button onClick={autoLabelAll} disabled={loading} style={{ background: "#4f46e5", color: "#fff" }}>🚀 Auto ALL (C++ Tracking)</button>
        <button onClick={trackToNext} disabled={loading} style={{ background: "#059669", color: "#fff" }}>➡️ Track</button>
        <button onClick={interpolate} style={{ background: "#db2777", color: "#fff", fontWeight: "bold" }}>↔️ Interpolate Keyframe</button>
        <button onClick={save} style={{ background: "#fff", color: "#000" }}>💾 Save</button>
        <button onClick={exportCOCO} style={{ background: "#f59e0b", color: "#000" }}>📥 COCO</button>
        <button onClick={() => setBoxes([])} style={{ color: "red" }}>Clear</button>
      </div>

      {jobId && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ height: 6, background: "#222", borderRadius: 4 }}>
            <div style={{ height: "100%", width: `${progress * 100}%`, background: "#00ff00" }} />
          </div>
          <div style={{ fontSize: 11, opacity: 0.6 }}>Job {jobId}: {Math.round(progress * 100)}%</div>
        </div>
      )}

      <div style={{ display: "flex", gap: 20 }}>
        <Stage
          width={960} height={540}
          onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}
          style={{ border: "2px solid #333", background: "#000", backgroundImage: `url(${API}/frames/${frameName})`, backgroundSize: "cover" }}
        >
          <Layer>
            {boxes.map((b, i) => (
              <Text key={`l-${i}`} x={b.x} y={b.y - 16} text={`${b.label} #${b.track_id} ${b.conf ? Math.round(b.conf * 100) + '%' : ''} ${b.interpolated ? '(interp)' : ''}`} fill="white" fontSize={12} />
            ))}
            {boxes.map((b, i) => (
              <Rect key={i} x={b.x} y={b.y} width={b.width} height={b.height}
                stroke={b.auto ? "#f59e0b" : b.tracked ? "#06b6d4" : b.interpolated ? "#db2777" : "lime"} strokeWidth={2} dash={b.auto ? [6, 4] : b.interpolated ? [3, 3] : []} />
            ))}
            {drawing && <Rect x={drawing.x} y={drawing.y} width={drawing.width} height={drawing.height} stroke="red" strokeWidth={2} dash={[5, 5]} />}
          </Layer>
        </Stage>

        <div style={{ width: 240 }}>
          <h4>Boxes ({boxes.length})</h4>
          {boxes.map((b, i) => (
            <div key={i} style={{ display: "flex", gap: 4, marginBottom: 6, alignItems: "center" }}>
              <input value={b.label} onChange={(e) => { const c = [...boxes]; c[i].label = e.target.value; setBoxes(c); }} style={{ width: 70 }} />
              <span style={{ fontSize: 10 }}>ID:{b.track_id}</span>
              <button onClick={() => setBoxes(boxes.filter((_, j) => j !== i))}>x</button>
            </div>
          ))}
          <div style={{ marginTop: 16, fontSize: 11, lineHeight: "18px", opacity: 0.7 }}>
            <div>🟩 Manual</div><div>🟧 Auto (C++ YOLO)</div><div>🟦 Tracked (C++ IOU)</div><div>🟪 Interpolated</div>
            <hr style={{ margin: "12px 0", opacity: 0.2 }} />
            <b>How to use Interpolation:</b><br />
            1. Draw box on frame 0<br />
            2. Click Interpolate (saves keyframe)<br />
            3. Go to frame 30, move box<br />
            4. Click Interpolate again -> fills 1-29
          </div>
          <div style={{ marginTop: 12 }}>
            <b>Keyframes:</b>
            {Object.entries(keyframes).map(([tid, kfs]) => (
              <div key={tid} style={{ fontSize: 11 }}>Track {tid}: {kfs.length} kfs at {kfs.map(k => k.frame_idx).join(",")}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}