import { useState } from 'react'
import Annotator from './Annotator.jsx'

export default function App() {
  const [videoId, setVideoId] = useState("demo")
  const [frames, setFrames] = useState([])
  const [videoUrl, setVideoUrl] = useState(null)
  const [uploaded, setUploaded] = useState(false)

  const handleUpload = async (e) => {
    const file = e.target.files[0]; if (!file) return
    const fd = new FormData(); fd.append('file', file)
    const r = await fetch('/upload-video', { method: 'POST', body: fd })
    const d = await r.json()
    setVideoUrl(d.video_url)
    // Fake 60 frames for demo (0,6,12...). In real app, extract frames with ffmpeg
    const fakeFrames = Array.from({ length: d.job_id ? 60 : 60 }, (_, i) => `frame_${i * 6}.jpg`)
    setFrames(fakeFrames)
    setUploaded(true)
  }

  return (
    <div style={{ padding: 16 }}>
      {!uploaded ? (
        <div style={{ textAlign: 'center', marginTop: 100 }}>
          <h1 style={{ color: '#0f0' }}>visionX VIDEO</h1>
          <input type="file" accept="video/*" onChange={handleUpload} />
          <p style={{ opacity: 0.5 }}>C++ Async + Tracking + Interpolation</p>
        </div>
      ) : (
        <Annotator videoId={videoId} frames={frames} videoUrl={videoUrl} />
      )}
    </div>
  )
}