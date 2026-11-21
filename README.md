# Virtual Hand Mouse

A real-time virtual mouse controlled via hand gestures using **Python**, **OpenCV**, **MediaPipe Hand Landmarker**, and **PyAutoGUI**, with camera input support (such as iVCam).

## Controls
- **Cursor Movement:** Index finger tracking inside the active region.
- **Left Click:** Extending/lifting the thumb with frame-stability confirmation.
- **Right Click:** Extending the pinky finger with frame-stability confirmation.
- **Exit:** Press `Q` in the display window.

## Setup & Requirements
1. Install dependencies:
   ```bash
   pip install opencv-python mediapipe numpy pyautogui
