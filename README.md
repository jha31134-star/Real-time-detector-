# Real-Time Object Detection Platform (cloud version)

YOLOv8 + OpenCV + MySQL + Streamlit, with the **browser webcam** streamed to the server via WebRTC.

## Deploy
1. Create a free hosted MySQL (Aiven / TiDB Cloud / similar). Create a database; the table is created automatically.
2. Push this folder to GitHub (`.env` is git-ignored).
3. Create a Hugging Face Space (Streamlit SDK) or a Streamlit Community Cloud app from the repo, main file `main.py`.
4. Add DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME as secrets / environment variables.
5. Open the app, click START, allow the camera.

Note: on some networks (college/office Wi-Fi) WebRTC needs a TURN server; if the video never starts, try another network.
