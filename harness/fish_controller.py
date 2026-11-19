import os
import time
import shutil
import queue
import logging
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler
import uvicorn

# --- CONFIGURATION ---
DATA_DIR = Path("/data")
CAPTURING_DIR = DATA_DIR / "capturing"
PENDING_DIR = DATA_DIR / "pending"
COMPLETED_DIR = DATA_DIR / "completed"

# Modes: "raw_video" (leave MP4s in /data/pending for laptop pull)
#        "local_cv"  (run CV script on each 15-min MP4, output CSV to /data/pending)
PIPELINE_MODE = "raw_video"
CV_SCRIPT_PATH = Path("/home/pi/fish_project/run_cv.py")
MIN_FREE_DISK_GB = 3.0  # Stop recording if SD card drops below 3 GB free
RPICAM_LOG_PATH = DATA_DIR / "rpicam_last.log"

last_session_id: str | None = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fishcam")

# --- GLOBAL STATE ---
analysis_queue = queue.Queue()
capture_lock = threading.Lock()
active_process: subprocess.Popen | None = None


def get_free_space_gb(folder: Path) -> float:
    total, used, free = shutil.disk_usage(folder)
    return free / (1024 ** 3)

def get_last_log_tail(lines: int = 15) -> str:
    if not RPICAM_LOG_PATH.exists():
        return "No log file yet."
    content = RPICAM_LOG_PATH.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])

# --- PROCESS 1: VIDEO CAPTURER ---
def start_recording_session(duration_ms: int = 28800000) -> str:
    global active_process, last_session_id

    with capture_lock:
        if active_process is not None and active_process.poll() is None:
            raise RuntimeError("A recording session is already in progress.")

        free_gb = get_free_space_gb(DATA_DIR)
        if free_gb < MIN_FREE_DISK_GB:
            raise RuntimeError(f"Insufficient disk space ({free_gb:.2f} GB free).")

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        last_session_id = session_id
        # Record raw .h264 elementary stream segments (required by --segment)
        output_pattern = str(CAPTURING_DIR / f"fish_{session_id}_%04d.h264")

        cmd = [
            "rpicam-vid",
            "-t", str(duration_ms),
            "--width", "1280",
            "--height", "720",
            "--framerate", "15",
            "--bitrate", "3500000",
            "--shutter", "10000",
            "--gain", "16.0",
            "--autofocus-mode", "manual",
            "--lens-position", "1.5",
            "--segment", "900000",
            "--inline",  # Crucial: writes SPS/PPS headers at every keyframe for clean splitting
            "-o", output_pattern,
            "--nopreview"
        ]

        logger.info(f"Starting capture session {session_id}: {' '.join(cmd)}")
        log_file = open(RPICAM_LOG_PATH, "w")
        active_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )

        time.sleep(1.5)
        if active_process.poll() is not None:
            exit_code = active_process.returncode
            error_tail = get_last_log_tail(10)
            logger.error(f"rpicam-vid exited immediately (code {exit_code}):\n{error_tail}")
            raise RuntimeError(f"rpicam-vid crashed (exit {exit_code}): {error_tail}")

        return session_id

def stop_recording_session() -> bool:
    """Gracefully stops rpicam-vid (SIGINT) so libav finalizes the current MP4."""
    global active_process
    with capture_lock:
        if active_process is None or active_process.poll() is not None:
            return False
        logger.info("Stopping active recording session gracefully...")
        active_process.terminate()
        try:
            active_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            active_process.kill()
        active_process = None
        return True


# --- BRIDGE: SEGMENT HARVESTER ---
def segment_harvester_loop():
    """
    Watches /data/capturing/ for completed .h264 segments,
    wraps them into .mp4 containers via ffmpeg (-c copy),
    and moves them to /data/pending/.
    """
    while True:
        try:
            h264_files = sorted(CAPTURING_DIR.glob("*.h264"), key=lambda p: p.stat().st_mtime)
            is_recording = (active_process is not None and active_process.poll() is None)

            if is_recording and get_free_space_gb(DATA_DIR) < MIN_FREE_DISK_GB:
                logger.error("Low disk space! Stopping recording session.")
                stop_recording_session()

            # All files except the newest are finished if recording; otherwise all are finished
            ready_files = h264_files[:-1] if is_recording else h264_files

            for src_h264 in ready_files:
                # 1. Wait until file hasn't been touched for 3 seconds
                if time.time() - src_h264.stat().st_mtime < 3.0:
                    continue

                # 2. Ignore 0-byte files or leftover literal '%04d' files
                if src_h264.stat().st_size == 0 or "%" in src_h264.name:
                    logger.warning(f"Removing invalid/empty file: {src_h264.name}")
                    src_h264.unlink()
                    continue

                temp_mp4 = src_h264.with_suffix(".mp4")
                dest_mp4 = PENDING_DIR / temp_mp4.name

                # 3. Explicitly specify '-f h264' so ffmpeg never guesses image sequences
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-f", "h264",
                        "-r", "15",
                        "-i", str(src_h264),
                        "-c", "copy",
                        str(temp_mp4)
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )

                src_h264.unlink()
                shutil.move(str(temp_mp4), str(dest_mp4))
                logger.info(f"Segment wrapped and moved to pending: {dest_mp4.name}")
                analysis_queue.put(dest_mp4)
        except Exception as e:
            logger.error(f"Error in segment harvester: {e}")

        time.sleep(10)

# --- PROCESS 2: BACKGROUND ANALYZER / OFFLOADER ---
def analyzer_worker_loop():
    """Consumes finalized 15-minute MP4 segments one at a time."""
    while True:
        video_path: Path = analysis_queue.get()
        try:
            if not video_path.exists():
                continue

            if PIPELINE_MODE == "raw_video":
                # Leave the finalized MP4 in /data/pending/ for rsync or Syncthing to pull
                logger.info(f"[raw_video mode] {video_path.name} ready in {PENDING_DIR} for transfer.")

            elif PIPELINE_MODE == "local_cv":
                csv_out = PENDING_DIR / f"{video_path.stem}.csv"
                logger.info(f"[local_cv mode] Running CV analysis on {video_path.name}...")

                subprocess.run(
                    ["python3", str(CV_SCRIPT_PATH), "--input", str(video_path), "--output", str(csv_out)],
                    check=True
                )

                # Once CSV is created in /data/pending/, delete or archive the raw MP4
                video_path.unlink()
                logger.info(f"Finished CV analysis: {csv_out.name} created, raw MP4 deleted.")

        except subprocess.CalledProcessError as e:
            logger.error(f"CV script failed on {video_path.name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected worker error on {video_path.name}: {e}")
        finally:
            analysis_queue.task_done()


# --- LIFESPAN, SCHEDULER & HTTP API ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in (CAPTURING_DIR, PENDING_DIR, COMPLETED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Start background daemon threads
    threading.Thread(target=segment_harvester_loop, daemon=True).start()
    threading.Thread(target=analyzer_worker_loop, daemon=True).start()

    # Load any leftover files in /data/pending/ on boot (if in local_cv mode)
    if PIPELINE_MODE == "local_cv":
        for leftover in sorted(PENDING_DIR.glob("*.mp4")):
            analysis_queue.put(leftover)

    # Optional scheduled daily start (e.g., every day at 08:00 AM for 8 hours)
    scheduler = BackgroundScheduler()
    scheduler.add_job(start_recording_session, "cron", hour=8, minute=0)
    scheduler.start()

    yield
    scheduler.shutdown()
    stop_recording_session()

app = FastAPI(lifespan=lifespan)

@app.get("/status")
def status():
    is_recording = (active_process is not None and active_process.poll() is None)
    exit_code = None if is_recording or active_process is None else active_process.returncode
    return {
        "recording": is_recording,
        "last_session_id": last_session_id,
        "last_exit_code": exit_code,
        "rpicam_log_tail": get_last_log_tail(10),
        "pipeline_mode": PIPELINE_MODE,
        "free_disk_gb": round(get_free_space_gb(DATA_DIR), 2),
        "capturing_count": len(list(CAPTURING_DIR.glob("*.h264"))),
        "pending_files": sorted([p.name for p in PENDING_DIR.iterdir()]),
        "analysis_queue_size": analysis_queue.qsize(),
    }

@app.post("/start")
def manual_start(hours: float = 8.0):
    try:
        duration_ms = int(hours * 3600 * 1000)
        session_id = start_recording_session(duration_ms=duration_ms)
        return {"status": "started", "session_id": session_id, "hours": hours}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stop")
def manual_stop():
    stopped = stop_recording_session()
    return {"status": "stopped" if stopped else "not_running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
