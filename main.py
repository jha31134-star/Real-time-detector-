"""Cloud version: the BROWSER's webcam is streamed to the server with WebRTC.
Run locally to test with:  python -m streamlit run main.py"""
import threading
import time

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

import config
import ui
from database import Database, DBWriter
from detector import Detector

st.set_page_config(page_title="Object Detection Platform", layout="wide")

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


@st.cache_resource
def get_detector(name):
    return Detector(name)


@st.cache_resource
def get_database():
    return Database(config.DB_CONFIG)


@st.cache_resource
def get_writer():
    return DBWriter(config.DB_CONFIG)


class Processor(VideoProcessorBase):
    """Runs in a background thread for every incoming webcam frame."""

    def __init__(self):
        self.detector = None
        self.writer = None
        self.conf, self.targets, self.imgsz, self.skip = 0.5, [], 416, 2
        self.dets, self.n, self.fps = [], 0, 0.0
        self.prev = time.time()
        self.last_logged = {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.detector is not None:
            if self.n % self.skip == 0:
                self.dets = self.detector.detect(img, self.conf, self.targets, self.imgsz)
                now = time.time()
                for d in self.dets:
                    if now - self.last_logged.get(d["object_class"], 0) >= config.LOG_COOLDOWN_SEC:
                        self.writer.submit(d)
                        self.last_logged[d["object_class"]] = now
            self.detector.draw(img, self.dets)
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.prev, 1e-6)
        self.prev = now
        self.n += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    _, model_name, conf, imgsz, skip = ui.sidebar_top(
        config.DEFAULT_CONFIDENCE, config.DEFAULT_IMGSZ, config.DEFAULT_FRAME_SKIP, show_source=False)
    detector = get_detector(model_name)
    targets = ui.class_picker(detector.class_names)

    db, writer = get_database(), get_writer()
    if not db.ok:
        db.connect()

    metrics, counts, feed, table = ui.layout()
    if not db.ok:
        st.warning(f"Database offline: {db.error}")
    ui.show_table(table, db.recent())
    ui.show_metrics(metrics, counts, 0, [], writer.written, db.ok)

    with feed.container():
        ctx = webrtc_streamer(
            key="object-detection", mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            video_processor_factory=Processor, async_processing=True)

    if ctx.video_processor:
        p = ctx.video_processor
        p.detector, p.writer = detector, writer
        p.conf, p.targets, p.imgsz, p.skip = conf, targets, imgsz, skip
        last_table = 0.0
        while ctx.state.playing:               # live dashboard numbers beside the video
            ui.show_metrics(metrics, counts, p.fps, p.dets, writer.written, db.ok)
            if time.time() - last_table > 1.5:
                last_table = time.time()
                if not db.ok:
                    db.connect()
                ui.show_table(table, db.recent())
            time.sleep(0.3)
    else:
        st.info("Click **START** under the video, then allow camera access in your browser.")


main()
