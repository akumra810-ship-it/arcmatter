# Steps to Execute
### 1. Get ONNX model
pip install ultralytics
yolo export model=yolov8n.pt format=onnx

### 2. Build C++ engine
mkdir backend/build && cd backend/build
cmake.. && make -j
cp visionx_engine*.so../
cd../..

### 3. Run
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

### 4. Open http://localhost:8000/frontend/
Upload mp4 -> C++ processes in background -> watch progress bar -> video with boxes

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

That's it. For real video annotation you need one more step: *tracking* - giving same car same ID across frames, and *interpolation* - if you label frame 0 and frame 30, fill boxes between them automatically.

# Code Logic -- visionx_engine.cpp
The provided code is a C++ implementation of an asynchronous video processing engine that utilizes the OpenCV library for computer vision tasks, specifically for object detection using a YOLO (You Only Look Once) model. The code is structured into several key components, including data structures for detections, video jobs, and tracking, as well as a main class `AsyncVideoEngine` that encapsulates the functionality for loading models, processing videos, and managing detection results.

### Data Structures
The code defines several structures to represent essential data. The `Detection` struct holds information about detected objects, including their position (x, y), size (width, height), class ID, confidence score, class name, frame index, and a tracking ID. The `VideoJob` struct manages the state of video processing jobs, including the job ID, video path, total frames, processed frames, all detections, and a completion flag. The `Track` struct is used for tracking detected objects across frames, storing the last detection, the number of frames since the object was last seen, and a unique track ID.

### AsyncVideoEngine Class
The `AsyncVideoEngine` class is the core of the implementation. It contains a private member for the YOLO model (`cv::dnn::Net net`), a map to store video jobs, and a mutex for thread safety. The class provides several public methods: `load_model` for loading the YOLO model from an ONNX file, `submit_video` for starting the video processing in a separate thread, `get_video_status` for retrieving the status of a video job, `get_video_results` for fetching detection results, and `interpolate_keyframes` for generating intermediate detections between keyframes.

### Object Detection and Tracking
The `run_yolo` method processes an image frame to detect objects using the YOLO model. It prepares the image as a blob, performs a forward pass through the network, and extracts bounding boxes, confidence scores, and class IDs for detected objects. The `iou` method calculates the Intersection over Union (IoU) between two detections, which is crucial for tracking objects across frames. The `assign_track_ids_internal` method manages the assignment of unique track IDs to detected objects, ensuring that the same object is tracked consistently across frames.

### Multithreading and Synchronization
The `submit_video` method initiates video processing in a separate thread, allowing the main application to remain responsive. It captures frames from the video, processes them at a specified frame rate, and updates the job status in a thread-safe manner using mutex locks. The use of `std::lock_guard` ensures that access to shared resources is synchronized, preventing data races.

### Python Bindings
Finally, the code uses the Pybind11 library to expose the `Detection` and `AsyncVideoEngine` classes to Python, allowing users to interact with the C++ implementation from Python scripts. This integration enables the use of the video processing engine in Python applications, leveraging the performance of C++ while providing a user-friendly interface.

Overall, this code represents a robust framework for real-time object detection in videos, combining advanced computer vision techniques with efficient multithreading and Python interoperability.

# Code logic - main.py
The `dict_to_det` function is designed to convert a dictionary representation of detection data into an instance of the `Detection` class from the `visionx_engine` module. This function takes two parameters: `d`, which is a dictionary containing the detection attributes, and `frame_idx`, which indicates the frame index associated with the detection.

Inside the function, a new `Detection` object is instantiated. The attributes of this object are populated using values extracted from the dictionary `d`. The `get` method is used to safely retrieve values, providing default values (like `0` for numerical attributes) in case the expected keys are not present. For instance, the position (`x`, `y`), dimensions (`w`, `h`), and identification attributes (`class_id`, `class_name`, `track_id`) are all set based on the dictionary's contents. This approach ensures that even if some data is missing, the function will not raise an error and will instead use sensible defaults.

The `InterpReq` class, which inherits from `BaseModel`, defines the structure of the request body expected by the `/interpolate` endpoint. It specifies that the request must contain two dictionaries: `start_box` and `end_box`, which represent the starting and ending detection boxes, respectively. Additionally, it requires two integers: `start_frame` and `end_frame`, which indicate the frames between which interpolation should occur. By using Pydantic's `BaseModel`, this class provides automatic data validation and serialization, ensuring that incoming requests conform to the expected format, thus enhancing the robustness of the API.