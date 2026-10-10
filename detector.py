"""YOLO detection: class filter + confidence threshold + per-class coloured boxes."""
import hashlib
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)            # downloads the weights on first use
        self.names = self.model.names            # {id: name}
        self.class_names = list(self.names.values())
        self.name_to_id = {v: k for k, v in self.names.items()}
        self.use_gpu = torch.cuda.is_available()
        self.device = 0 if self.use_gpu else "cpu"
        self._colors = {}
        # warm-up so the first live frame isn't slow
        self.model.predict(np.zeros((320, 320, 3), np.uint8), imgsz=320,
                           device=self.device, verbose=False)

    def detect(self, frame, conf: float, target_classes=None, imgsz: int = 416):
        """Returns a list of dicts: object_class, confidence, bbox_x/y/w/h."""
        ids = None
        if target_classes:
            ids = [self.name_to_id[n] for n in target_classes if n in self.name_to_id]
            if not ids:
                return []
        r = self.model.predict(frame, conf=conf, imgsz=imgsz, classes=ids, max_det=100,
                               device=self.device, half=self.use_gpu, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                dets.append({"object_class": self.names[int(k)], "confidence": float(c),
                             "bbox_x": int(x1), "bbox_y": int(y1),
                             "bbox_w": int(x2 - x1), "bbox_h": int(y2 - y1)})
        return dets

    def _color(self, label: str):
        if label not in self._colors:
            h = hashlib.md5(label.encode()).digest()
            self._colors[label] = (80 + h[0] % 176, 80 + h[1] % 176, 80 + h[2] % 176)
        return self._colors[label]

    def draw(self, frame, dets):
        for d in dets:
            x1, y1 = d["bbox_x"], d["bbox_y"]
            x2, y2 = x1 + d["bbox_w"], y1 + d["bbox_h"]
            col = self._color(d["object_class"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            text = f'{d["object_class"]} {d["confidence"] * 100:.0f}%'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            top = max(y1 - th - 8, 0)
            cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + 8), col, -1)
            cv2.putText(frame, text, (x1 + 3, top + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return frame
