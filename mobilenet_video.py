# import cv2, tensorflow as tf, numpy as np, time

# interpreter = tf.lite.Interpreter(
#     model_path="ssd_mobilenet_v2_coco_quant_postprocess.tflite")
# interpreter.allocate_tensors()

# in_det = interpreter.get_input_details()
# out_det = interpreter.get_output_details()

# cap = cv2.VideoCapture("input1.mp4")
# w = int(cap.get(3))
# h = int(cap.get(4))
# fps = int(cap.get(5))

# out = cv2.VideoWriter("mobilenet_output.mp4",
#                       cv2.VideoWriter_fourcc(*"mp4v"),
#                       fps, (w,h))

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     img = cv2.resize(frame,(300,300))
#     img = np.expand_dims(img,0)

#     start = time.time()
#     interpreter.set_tensor(in_det[0]['index'], img)
#     interpreter.invoke()
#     fps_live = 1/(time.time()-start)

#     boxes = interpreter.get_tensor(out_det[0]['index'])[0]
#     classes = interpreter.get_tensor(out_det[1]['index'])[0]
#     scores = interpreter.get_tensor(out_det[2]['index'])[0]

#     for i in range(len(scores)):
#         if scores[i] > 0.3 and int(classes[i]) == 0:
#             y1,x1,y2,x2 = boxes[i]
#             x1,x2 = int(x1*w), int(x2*w)
#             y1,y2 = int(y1*h), int(y2*h)

#             cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
#             cv2.putText(frame,
#                 f"person {scores[i]:.2f}",
#                 (x1,y1-10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,(255,0,0),2)

#     cv2.putText(frame,
#         f"MobileNet FPS: {int(fps_live)}",
#         (20,40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,(0,0,255),2)

#     out.write(frame)

# cap.release()
# out.release()
import cv2
import tensorflow as tf
import numpy as np
import time

interpreter = tf.lite.Interpreter(
    model_path="ssd_mobilenet_v2_coco_quant_postprocess.tflite")
interpreter.allocate_tensors()

in_det = interpreter.get_input_details()
out_det = interpreter.get_output_details()

cap = cv2.VideoCapture("input1.mp4")
w = int(cap.get(3))
h = int(cap.get(4))
fps = int(cap.get(5))

out = cv2.VideoWriter(
    "mobilenet_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame,(300,300))
    img = np.expand_dims(img,0)

    start = time.time()
    interpreter.set_tensor(in_det[0]['index'], img)
    interpreter.invoke()
    fps_live = 1/(time.time()-start)

    boxes = interpreter.get_tensor(out_det[0]['index'])[0]
    classes = interpreter.get_tensor(out_det[1]['index'])[0]
    scores = interpreter.get_tensor(out_det[2]['index'])[0]

    human_count = 0

    for i in range(len(scores)):
        if scores[i] > 0.5 and int(classes[i]) == 0:
            human_count += 1
            y1,x1,y2,x2 = boxes[i]
            x1,x2 = int(x1*w), int(x2*w)
            y1,y2 = int(y1*h), int(y2*h)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

    cv2.putText(frame,
        f"MobileNet Humans: {human_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.putText(frame,
        f"FPS: {int(fps_live)}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    out.write(frame)

cap.release()
out.release()
print("MobileNet video created")

