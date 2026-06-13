import cv2
import numpy as np
import os
import pygame
import time
from PIL import ImageFont, ImageDraw, Image

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load Kannada labels
with open("coco_kannada.names", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]
print("Loaded labels count:", len(classes))

# Path to Kannada font
kannada_font_path = "NotoSansKannada-Regular.ttf"

# Track which labels have already been spoken
spoken_labels = set()

# Function to draw Kannada text using PIL
def draw_kannada_text_pil(frame, text, position, font_size=28):
    cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)

    try:
        font = ImageFont.truetype(kannada_font_path, font_size)
    except Exception as e:
        print("Font load error:", e)
        return frame

    draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

# Object detection function
def detect_objects(frame):
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confs, cids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            if scores.size == 0:
                continue
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf > 0.5:
                cx, cy = int(det[0] * W), int(det[1] * H)
                bw, bh = int(det[2] * W), int(det[3] * H)
                x, y = cx - bw // 2, cy - bh // 2
                boxes.append([x, y, bw, bh])
                confs.append(conf)
                cids.append(cid)

    idxs = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    return boxes, confs, cids, idxs.flatten() if len(idxs) > 0 else []

# Start webcam
cap = cv2.VideoCapture(0)

# Audio folder
audio_folder = "audio_clips"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    boxes, confs, cids, idxs = detect_objects(frame)

    for i in idxs:
        if i >= len(boxes) or cids[i] >= len(classes):
            continue

        label = classes[cids[i]]
        confidence = confs[i]
        box = boxes[i]

        # Draw bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)

        # Label with confidence
        label_with_conf = f"{label} ({int(confidence * 100)}%)"
        frame = draw_kannada_text_pil(frame, label_with_conf, (box[0], max(0, box[1] - 30)))

        # Play audio only once per label per session
        if label not in spoken_labels:
            audio_path = os.path.join(audio_folder, f"{label}.wav")
            if os.path.exists(audio_path):
                try:
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print(f"Audio playback error for '{label}':", e)
            else:
                print(f"Audio file missing for: {label}")

            spoken_labels.add(label)

    cv2.imshow("YOLOv3 Kannada Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
