bash# 1. Get ONNX model
pip install ultralytics
yolo export model=yolov8n.pt format=onnx

# 2. Build C++ engine
mkdir backend/build && cd backend/build
cmake.. && make -j
cp visionx_engine*.so../
cd../..

# 3. Run
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 4. Open http://localhost:8000/frontend/
# Upload mp4 -> C++ processes in background -> watch progress bar -> video with boxes

# visionx_engine logic
### 1. The Engine Starts - Like Opening a Factory

When Python says `load_model("yolov8n.onnx")`, C++ opens the YOLO brain file and keeps it in memory. Think of it like loading a worker who already knows how to spot cars and people. This happens once, at startup. After that, the factory is ready.

### 2. When You Upload a Video

You upload a 1-minute video. That video is 30 frames per second, so 1800 images total.

C++ doesn't process all 1800. It says: "I will only check 5 frames per second, because frames next to each other look almost same."

So 1 minute x 5 fps = 300 frames to process. That's the job.

C++ creates a Job Ticket with an ID like `job_12345` and says to Python: "I took your video, here is your ticket. Go check progress later, don't wait here." Python returns immediately, so your website doesn't freeze.

### 3. The Background Workers - The Real Magic

Inside C++, there is a separate invisible thread. Think of it as a worker in the back room.

This worker does this loop for your video:

- Open video file
- Read frame 0, skip frame 1-5, read frame 6, skip 7-11, read frame 12... (because 30fps / 5fps = step of 6)
- For each frame it reads, it runs YOLO:
  a) Resize image to 640x640 (YOLO needs small image)
  b) Normalize it (divide by 255)
  c) Send it through the neural network (this is `net.forward()` - the brain thinks)
  d) Brain returns 8400 guesses: "I think there is a car at x,y with 90% confidence"
  e) Filter guesses: keep only if confidence > 50%
  f) Remove duplicates: if two boxes overlap same car, keep the best one (this is NMS)
  g) Save the final boxes with frame number

After finishing one frame, it updates a counter: "Processed 50/300 frames". Python can read this counter anytime to show your progress bar.

### 4. Why It's Async and Fast

*Sync (your old version):* Python says "process this frame" and waits. C++ processes. Python waits 200ms doing nothing. If you have 300 frames, Python waits 300 x 200ms = 60 seconds, blocked. No one else can use the website.

*Async (this new version):*
- Python says "process whole video" and gets ticket in 1ms, free to serve other users
- C++ worker processes all 300 frames in background, using all CPU cores
- Frontend keeps asking "how much done?" every 1 second - gets progress
- When done, C++ marks job as done: `done = true`
- Frontend then asks "give me all results"

It's like restaurant: You order (submit video), get token (job_id), sit down (polling). Kitchen (C++ thread) cooks in background. You don't stand in kitchen waiting.

### 5. The Results

When done, C++ has an array like:
- Frame 0: [car at x,y, person at x,y]
- Frame 6: [car at x,y, person at x,y]
- Frame 12: [car at x,y]

Frontend then draws these boxes on top of your video. When you play video, it shows boxes for the nearest sampled frame.

That's it. For real video annotation you need one more step: *tracking* - giving same car same ID across frames, and *interpolation* - if you label frame 0 and frame 30, fill boxes between them automatically. Want me to explain that logic too?