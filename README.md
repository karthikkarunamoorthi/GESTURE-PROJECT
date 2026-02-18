# Gesture Detection

Touchless media controls using MediaPipe hand tracking and OpenCV. Gestures drive play/pause/track changes plus a volume mode.

## Prerequisites
- Windows 10/11 with a working webcam
- Python 3.10+ (64-bit recommended)
- VLC or your default media player responding to system media keys

## Setup (Windows)
1) Create and activate a virtual env (optional):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # or use Activate.bat in cmd
python -m pip install --upgrade pip
```
2) Install dependencies:
```powershell
pip install -r requirements.txt
```
3) Download the MediaPipe model (saves to `models/hand_landmarker.task`):
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models/hand_landmarker.task"
```

## Running
```powershell
python main.py
```
- Grant webcam permission when prompted.
- Press `q` in the preview window to exit.

## Gestures
- 5 fingers: play
- Fist (0): pause
- Thumb up only (1): previous
- Three fingers (3): next
- Hold ✌ (2) for ~1s: enter volume mode
  - Fingers spread (>0.16 normalized distance): volume up
  - Fingers closer: volume down
  - Make a fist while in volume mode to exit

## Notes
- On Windows, media and volume actions use system media keys via `pyautogui`. On Linux, the script still calls `playerctl`/`pamixer` if present.
- If the model file is missing, the script raises an error with the expected path.
- Adjust camera index in `main.py` if your default webcam is not at index 0.

--
Project: Gesture Detection