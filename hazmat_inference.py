import cv2
from ultralytics import YOLO

_MODEL = None
_DEVICE = None


def init_inference(weights_path="/home/student/documents/rmrc_hazmat/yoloversion/best.pt", device_str="cpu"):
    global _MODEL, _DEVICE

    _DEVICE = device_str
    _MODEL = YOLO(weights_path)


def run_frame(frame, confidence_threshold=0.4):
    if _MODEL is None:
        raise RuntimeError("Inference not initialized. Call init_inference() first.")

    annotated_frame = frame.copy()
    detected_labels = []

    results = _MODEL.predict(
        source=frame,
        conf=confidence_threshold,
        device=_DEVICE,
        verbose=False
    )

    result = results[0]

    if result.boxes is None:
        return annotated_frame, detected_labels

    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        detected_labels.append(label)

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label_text = f"{label} {conf:.2f}"

        cv2.putText(
            annotated_frame,
            label_text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return annotated_frame, detected_labels