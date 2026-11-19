import io
import os
import time
import threading
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
import uvicorn

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, MJPEGEncoder
from picamera2.outputs import FileOutput, FfmpegOutput
from libcamera import controls

app = FastAPI()
picam2 = Picamera2()
camera_lock = threading.Lock()
TEST_CLIP_PATH = Path("/tmp/test_clip.mp4")

# Current state
state = {
    "width": 1280,
    "height": 720,
    "fps": 15,
    "bitrate_mbps": 2.5,
    "lens_position": 2.0,     # Diopters (0 = infinity, 2.0 = 50cm)
    "shutter_us": 10000,      # 10ms (1/100s) prevents motion blur
    "gain": 2.0,              # ISO sensitivity equivalent
    "auto_exposure": False,
}

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

stream_output = StreamingOutput()
mjpeg_encoder = None

def apply_camera_controls():
    ctrl = {"FrameRate": state["fps"]}
    # Only set LensPosition if camera supports autofocus (e.g., Pi Camera Module 3)
    if "LensPosition" in picam2.camera_controls:
        ctrl["AfMode"] = controls.AfModeEnum.Manual
        ctrl["LensPosition"] = float(state["lens_position"])
    if state["auto_exposure"]:
        ctrl["AeEnable"] = True
    else:
        ctrl["AeEnable"] = False
        ctrl["ExposureTime"] = int(state["shutter_us"])
        ctrl["AnalogueGain"] = float(state["gain"])
    picam2.set_controls(ctrl)

def start_preview_stream():
    global mjpeg_encoder
    if picam2.started:
        picam2.stop_recording()
    config = picam2.create_video_configuration(
        main={"size": (state["width"], state["height"]), "format": "YUV420"}
    )
    picam2.configure(config)
    apply_camera_controls()
    mjpeg_encoder = MJPEGEncoder(bitrate=10_000_000)
    picam2.start_recording(mjpeg_encoder, FileOutput(stream_output))

@app.on_event("startup")
def startup():
    with camera_lock:
        start_preview_stream()

@app.get("/stream.mjpg")
def video_feed():
    def generate():
        while True:
            with stream_output.condition:
                stream_output.condition.wait()
                frame = stream_output.frame
            yield (b"--FRAME\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=FRAME")

@app.post("/update")
def update_settings(
    res: str = Query("1280x720"),
    fps: int = Query(15),
    bitrate_mbps: float = Query(2.5),
    lens_position: float = Query(2.0),
    shutter_us: int = Query(10000),
    gain: float = Query(2.0),
    auto_exposure: bool = Query(False),
):
    w, h = map(int, res.split("x"))
    resolution_changed = (w != state["width"] or h != state["height"])
    state.update({
        "width": w, "height": h, "fps": fps, "bitrate_mbps": bitrate_mbps,
        "lens_position": lens_position, "shutter_us": shutter_us,
        "gain": gain, "auto_exposure": auto_exposure
    })
    with camera_lock:
        if resolution_changed:
            start_preview_stream()
        else:
            apply_camera_controls()
    return JSONResponse(state)

@app.post("/record_sample")
def record_sample():
    """Stops preview briefly, records 5s H.264 MP4 at target bitrate, restarts preview."""
    with camera_lock:
        if TEST_CLIP_PATH.exists():
            TEST_CLIP_PATH.unlink()
        picam2.stop_recording()

        config = picam2.create_video_configuration(
            main={"size": (state["width"], state["height"]), "format": "YUV420"}
        )
        picam2.configure(config)
        apply_camera_controls()

        bps = int(state["bitrate_mbps"] * 1_000_000)
        h264_encoder = H264Encoder(bitrate=bps)
        output = FfmpegOutput(str(TEST_CLIP_PATH))

        picam2.start_recording(h264_encoder, output)
        time.sleep(5)
        picam2.stop_recording()

        # Resume live preview
        start_preview_stream()

    size_bytes = TEST_CLIP_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    est_8hr_gb = (size_mb * (28800 / 5)) / 1024
    return {
        "clip_size_mb": round(size_mb, 2),
        "est_8hr_gb": round(est_8hr_gb, 2),
        "cli_cmd": (
            f"rpicam-vid -t 28800000 --width {state['width']} --height {state['height']} "
            f"--framerate {state['fps']} --bitrate {bps} --shutter {state['shutter_us']} "
            f"--gain {state['gain']} --autofocus-mode manual --lens-position {state['lens_position']} "
            f"--segment 900000 --codec libav -o /data/pending/fish_%04d.mp4 --nopreview"
        )
    }

@app.get("/sample.mp4")
def get_sample():
    return FileResponse(TEST_CLIP_PATH, media_type="video/mp4")

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>FishCam Parameter Tuner</title>
    <style>
        body { font-family: system-ui, sans-serif; margin: 20px; background: #18181b; color: #f4f4f5; }
        .grid { display: grid; grid-template-columns: 340px 1fr; gap: 20px; }
        .panel { background: #27272a; padding: 16px; border-radius: 8px; }
        .control { margin-bottom: 14px; }
        label { display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 4px; }
        input[type=range], select { width: 100%; }
        button { width: 100%; padding: 10px; background: #2563eb; color: white; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; margin-top: 8px; }
        button:hover { background: #1d4ed8; }
        .views { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        video, img { width: 100%; background: #000; border-radius: 6px; }
        pre { background: #09090b; padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 12px; color: #4ade80; }
    </style>
</head>
<body>
    <h2>FishCam Focus, Exposure & Bitrate Tuner</h2>
    <div class="grid">
        <div class="panel">
            <div class="control">
                <label>Resolution</label>
                <select id="res" onchange="sync()">
                    <option value="1280x720" selected>1280x720 (720p)</option>
                    <option value="1920x1080">1920x1080 (1080p)</option>
                    <option value="640x480">640x480 (VGA)</option>
                </select>
            </div>
            <div class="control">
                <label>Framerate (FPS) <span id="fps_val">15</span></label>
                <input type="range" id="fps" min="5" max="30" step="1" value="15" oninput="sync()">
            </div>
            <div class="control">
                <label>Target Bitrate (Mbps) <span id="br_val">2.5</span></label>
                <input type="range" id="bitrate" min="0.5" max="15" step="0.5" value="2.5" oninput="sync()">
            </div>
            <div class="control">
                <label>Focus (Diopters: 0=Inf, 2=50cm) <span id="focus_val">2.0</span></label>
                <input type="range" id="focus" min="0.0" max="10.0" step="0.1" value="2.0" oninput="sync()">
            </div>
            <div class="control">
                <label><input type="checkbox" id="ae" onchange="sync()"> Auto Exposure</label>
            </div>
            <div class="control">
                <label>Shutter Speed (µs) <span id="shutter_val">10000</span></label>
                <input type="range" id="shutter" min="1000" max="33000" step="500" value="10000" oninput="sync()">
            </div>
            <div class="control">
                <label>Analogue Gain (ISO) <span id="gain_val">2.0</span></label>
                <input type="range" id="gain" min="1.0" max="16.0" step="0.5" value="2.0" oninput="sync()">
            </div>
            <button id="recBtn" onclick="recordSample()">Record 5s H.264 Test Clip</button>
            <div id="stats" style="margin-top:12px; font-size:14px;"></div>
        </div>
        <div>
            <div class="views">
                <div class="panel">
                    <h4>Live Viewfinder (Focus & Exposure)</h4>
                    <img src="/stream.mjpg" />
                </div>
                <div class="panel">
                    <h4>Actual Encoded H.264 Output (Bitrate Check)</h4>
                    <video id="sampleVideo" controls loop></video>
                </div>
            </div>
            <div class="panel" style="margin-top:16px;">
                <h4>Generated Production CLI Command</h4>
                <pre id="cliBox">Click "Record 5s H.264 Test Clip" to measure real bitrate and generate command...</pre>
            </div>
        </div>
    </div>
<script>
let timer;
function sync() {
    document.getElementById('fps_val').innerText = document.getElementById('fps').value;
    document.getElementById('br_val').innerText = document.getElementById('bitrate').value;
    document.getElementById('focus_val').innerText = document.getElementById('focus').value;
    document.getElementById('shutter_val').innerText = document.getElementById('shutter').value;
    document.getElementById('gain_val').innerText = document.getElementById('gain').value;
    clearTimeout(timer);
    timer = setTimeout(sendUpdate, 150);
}
async function sendUpdate() {
    const params = new URLSearchParams({
        res: document.getElementById('res').value,
        fps: document.getElementById('fps').value,
        bitrate_mbps: document.getElementById('bitrate').value,
        lens_position: document.getElementById('focus').value,
        shutter_us: document.getElementById('shutter').value,
        gain: document.getElementById('gain').value,
        auto_exposure: document.getElementById('ae').checked
    });
    await fetch('/update?' + params.toString(), {method: 'POST'});
}
async function recordSample() {
    const btn = document.getElementById('recBtn');
    btn.disabled = true;
    btn.innerText = "Recording 5s H.264 clip...";
    const res = await fetch('/record_sample', {method: 'POST'});
    const data = await res.json();
    document.getElementById('sampleVideo').src = '/sample.mp4?t=' + Date.now();
    document.getElementById('sampleVideo').play();
    document.getElementById('stats').innerHTML = 
        `<b>5s Clip Size:</b> ${data.clip_size_mb} MB<br><b>Measured 8-Hr Total:</b> ${data.est_8hr_gb} GB`;
    document.getElementById('cliBox').innerText = data.cli_cmd;
    btn.disabled = false;
    btn.innerText = "Record 5s H.264 Test Clip";
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
