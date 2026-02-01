from ultralytics import YOLO
import cv2, time

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("input1.mp4")
w = int(cap.get(3))
h = int(cap.get(4))
fps = int(cap.get(5))

print("Opened:", cap.isOpened())

out = cv2.VideoWriter(
    "scylla_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    results = model(frame)
    fps_live = 1 / (time.time() - start)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if label == "person":
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,
                    f"{label} {conf:.2f}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(0,255,0),2)

    cv2.putText(frame,
        f"Scylla AI FPS: {int(fps_live)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,(0,0,255),2)

    out.write(frame)

cap.release()
out.release()
print("Scylla video created")
