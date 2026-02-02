# Tennis Rally Counter

This repository contains code to detect tennis balls in images and count rallies.

## Features
- Tennis Ball Detection (using YOLOv8)
- Tennis Court Line Detection

## Credits
The model and court line detector logic is adapted from [tennis_analysis](https://github.com/MuhammadMoinFaisal/tennis_analysis/tree/main).

## Getting Started

1.  Clone the repository
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the detection script:
    ```bash
    python detect_ball.py
    ```
4.  Run the video detection script:
    ```bash
    python detect_ball_video.py
    ```
