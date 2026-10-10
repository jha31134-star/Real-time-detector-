"""Streamlit dashboard components."""
import pandas as pd
import streamlit as st

MODELS = {
    "YOLOv8n - fastest": "yolov8n.pt",
    "YOLOv8s - more accurate (slower)": "yolov8s.pt",
}

PRESETS = {
    "All objects": [],
    "People & gadgets": ["person", "cell phone", "laptop", "keyboard", "mouse", "remote", "tv",
                         "book", "bottle", "cup", "backpack", "handbag", "clock", "scissors"],
    "Vehicles & road": ["bicycle", "car", "motorcycle", "bus", "truck", "train", "airplane",
                        "boat", "traffic light", "stop sign"],
    "Animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "Food & kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                       "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                       "donut", "cake"],
    "Custom": ["person"],
}
IMG_SIZES = [320, 416, 512, 640]


def sidebar_top(default_conf, default_imgsz, default_skip, show_source=True):
    st.sidebar.header("Settings")
    source = st.sidebar.text_input("Source (0 = webcam, or video path)", "0") if show_source else "0"
    model_label = st.sidebar.selectbox("Model", list(MODELS))
    conf = st.sidebar.slider("Confidence threshold (%)", 10, 100, default_conf)
    imgsz = st.sidebar.select_slider("Image size (smaller = faster)", options=IMG_SIZES,
                                     value=default_imgsz if default_imgsz in IMG_SIZES else 416)
    skip = st.sidebar.slider("Detect every N frames (higher = faster)", 1, 5, default_skip)
    return source, MODELS[model_label], conf / 100.0, imgsz, skip


def class_picker(class_names):
    preset = st.sidebar.selectbox("Object preset", list(PRESETS))
    if preset == "All objects":
        return []
    default = [n for n in PRESETS[preset] if n in class_names]
    return st.sidebar.multiselect("Objects to detect", class_names, default=default,
                                  key=f"cls_{preset}")


def run_toggle():
    return st.sidebar.toggle("Run detection", value=False)


def layout():
    """Creates the placeholders that main.py updates every frame."""
    st.title("Real-Time Object Detection Platform")
    c1, c2, c3, c4 = st.columns(4)
    metrics = {"fps": c1.empty(), "objects": c2.empty(),
               "logged": c3.empty(), "db": c4.empty()}
    counts = st.empty()
    feed = st.empty()
    st.subheader("Latest database logs")
    table = st.empty()
    return metrics, counts, feed, table


def show_metrics(metrics, counts, fps, dets, logged, db_ok):
    metrics["fps"].metric("FPS", f"{fps:.1f}")
    metrics["objects"].metric("Objects in frame", len(dets))
    metrics["logged"].metric("Logged this session", logged)
    metrics["db"].metric("MySQL", "Connected" if db_ok else "Offline")
    tally = {}
    for d in dets:
        tally[d["object_class"]] = tally.get(d["object_class"], 0) + 1
    counts.markdown("**Live count:** " + (" | ".join(f"{k} x{v}" for k, v in sorted(tally.items()))
                                           or "nothing detected"))


def show_table(table, rows):
    if rows:
        table.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        table.info("No logs yet.")
