#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <unordered_map>
namespace py = pybind11;
using namespace pybind11::literals;

struct Detection
{
    int x, y, w, h;
    int class_id;
    float conf;
    std::string class_name;
    int frame_idx;
    int track_id = 0;
};
struct VideoJob
{
    std::string job_id;
    std::string video_path;
    int total_frames = 0;
    int processed = 0;
    std::vector<std::vector<Detection>> all_detections;
    bool done = false;
};

struct Track
{
    int track_id;
    Detection last_det;
    int frames_since_seen = 0;
};

class AsyncVideoEngine
{
private:
    cv::dnn::Net net;
    std::unordered_map<std::string, VideoJob> jobs;
    std::mutex job_mtx;
    std::vector<std::string> names = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    // ---- YOLO ----
    std::vector<Detection> run_yolo(cv::Mat &img, int frame_idx)
    {
        int W = img.cols, H = img.rows;
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());
        cv::Mat ot;
        cv::transpose(outs[0].reshape(1, 84), ot);
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        std::vector<int> cids;
        for (int i = 0; i < ot.rows; i++)
        {
            float *d = (float *)ot.row(i).data;
            cv::Mat s(1, 80, CV_32F, d + 4);
            cv::Point id;
            double sc;
            cv::minMaxLoc(s, 0, &sc, 0, &id);
            if (sc > 0.5)
            {
                boxes.push_back(cv::Rect(int((d[0] - d[2] / 2) * W / 640), int((d[1] - d[3] / 2) * H / 640), int(d[2] * W / 640), int(d[3] * H / 640)));
                confs.push_back(sc);
                cids.push_back(id.x);
            }
        }
        std::vector<int> idx;
        cv::dnn::NMSBoxes(boxes, confs, 0.5, 0.45, idx);
        std::vector<Detection> res;
        for (int i : idx)
            res.push_back({boxes[i].x, boxes[i].y, boxes[i].width, boxes[i].height, cids[i], confs[i], names[cids[i]], frame_idx, 0});
        return res;
    }

    // ---- IOU ----
    float iou(Detection &a, Detection &b)
    {
        int x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.w, b.x + b.w), y2 = std::min(a.y + a.h, b.y + b.h);
        int iw = std::max(0, x2 - x1), ih = std::max(0, y2 - y1);
        float inter = iw * ih;
        float uni = a.w * a.h + b.w * b.h - inter;
        return uni > 0 ? inter / uni : 0;
    }

    // ---- TRACKING ----
    void assign_track_ids_internal(std::vector<std::vector<Detection>> &all_frames)
    {
        std::vector<Track> tracks;
        int next_id = 1;
        for (int f = 0; f < all_frames.size(); f++)
        {
            for (auto &t : tracks)
                t.frames_since_seen++;
            for (auto &det : all_frames[f])
            {
                int best = -1;
                float best_iou = 0.3;
                for (int i = 0; i < tracks.size(); i++)
                {
                    if (tracks[i].last_det.class_id != det.class_id)
                        continue;
                    if (tracks[i].frames_since_seen > 30)
                        continue;
                    float cur = iou(det, tracks[i].last_det);
                    if (cur > best_iou)
                    {
                        best_iou = cur;
                        best = i;
                    }
                }
                if (best != -1)
                {
                    det.track_id = tracks[best].track_id;
                    tracks[best].last_det = det;
                    tracks[best].frames_since_seen = 0;
                }
                else
                {
                    det.track_id = next_id++;
                    tracks.push_back({det.track_id, det, 0});
                }
            }
        }
    }

public:
    bool load_model(const std::string &path)
    {
        net = cv::dnn::readNetFromONNX(path);
        return true;
    }

    std::string submit_video(const std::string &video_path, int sample_fps = 5)
    {
        std::string job_id = "job_" + std::to_string(rand() % 100000);
        {
            std::lock_guard<std::mutex> lock(job_mtx);
            jobs[job_id] = {job_id, video_path, 0, 0, {}, false};
        }
        std::thread([this, job_id, video_path, sample_fps]
                    {
            cv::VideoCapture cap(video_path); double fps=cap.get(cv::CAP_PROP_FPS); int total=cap.get(cv::CAP_PROP_FRAME_COUNT);
            int step=std::max(1,(int)(fps/sample_fps)); int out_total=total/step;
            { std::lock_guard<std::mutex> lock(job_mtx); jobs[job_id].total_frames=out_total; jobs[job_id].all_detections.resize(out_total); }
            cv::Mat frame; int f_idx=0,out_idx=0;
            while(cap.read(frame)){
                if(f_idx%step==0){
                    auto dets=run_yolo(frame,f_idx);
                    std::lock_guard<std::mutex> lock(job_mtx);
                    if(jobs.count(job_id)){ jobs[job_id].all_detections[out_idx]=dets; jobs[job_id].processed++; }
                    out_idx++;
                }
                f_idx++;
            }
            // Tracking after all frames
            { std::lock_guard<std::mutex> lock(job_mtx);
              if(jobs.count(job_id)){ assign_track_ids_internal(jobs[job_id].all_detections); jobs[job_id].done=true; }
            } })
            .detach();
        return job_id;
    }

    py::dict get_video_status(const std::string &job_id)
    {
        std::lock_guard<std::mutex> lock(job_mtx);
        if (!jobs.count(job_id))
        {
            py::dict d;
            d["error"] = "not found";
            return d;
        }
        auto &j = jobs[job_id];
        py::dict d;
        d["job_id"] = j.job_id;
        d["total"] = j.total_frames;
        d["processed"] = j.processed;
        d["progress"] = j.total_frames > 0 ? (float)j.processed / j.total_frames : 0;
        d["done"] = j.done;
        return d;
    }

    std::vector<std::vector<Detection>> get_video_results(const std::string &job_id)
    {
        std::lock_guard<std::mutex> lock(job_mtx);
        if (!jobs.count(job_id))
            return {};
        return jobs[job_id].all_detections;
    }

    // ---- INTERPOLATION API ----
    std::vector<Detection> interpolate_keyframes(Detection start_box, Detection end_box, int start_frame, int end_frame)
    {
        std::vector<Detection> result;
        int n = end_frame - start_frame;
        if (n <= 0)
            return {start_box};
        for (int i = 0; i <= n; i++)
        {
            float t = (float)i / n;
            Detection d;
            d.x = start_box.x + (end_box.x - start_box.x) * t;
            d.y = start_box.y + (end_box.y - start_box.y) * t;
            d.w = start_box.w + (end_box.w - start_box.w) * t;
            d.h = start_box.h + (end_box.h - start_box.h) * t;
            d.class_id = start_box.class_id;
            d.class_name = start_box.class_name;
            d.track_id = start_box.track_id;
            d.frame_idx = start_frame + i;
            d.conf = 1.0;
            result.push_back(d);
        }
        return result;
    }
};

PYBIND11_MODULE(visionx_engine, m)
{
    py::class_<Detection>(m, "Detection")
        .def(py::init<>())
        .def_readwrite("x", &Detection::x)
        .def_readwrite("y", &Detection::y)
        .def_readwrite("w", &Detection::w)
        .def_readwrite("h", &Detection::h)
        .def_readwrite("class_id", &Detection::class_id)
        .def_readwrite("conf", &Detection::conf)
        .def_readwrite("class_name", &Detection::class_name)
        .def_readwrite("frame_idx", &Detection::frame_idx)
        .def_readwrite("track_id", &Detection::track_id);
    py::class_<AsyncVideoEngine>(m, "AsyncVideoEngine")
        .def(py::init<>())
        .def("load_model", &AsyncVideoEngine::load_model)
        .def("submit_video", &AsyncVideoEngine::submit_video)
        .def("get_video_status", &AsyncVideoEngine::get_video_status)
        .def("get_video_results", &AsyncVideoEngine::get_video_results)
        .def("interpolate_keyframes", &AsyncVideoEngine::interpolate_keyframes);
}