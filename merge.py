import cv2, numpy as np

v1 = cv2.VideoCapture("raw1_output.mp4")
v2 = cv2.VideoCapture("yolo_output.mp4")
v3 = cv2.VideoCapture("mobilenet_output.mp4")
v4 = cv2.VideoCapture("scylla_output.mp4")

w = int(v1.get(3))
h = int(v1.get(4))
fps = int(v1.get(5))

out = cv2.VideoWriter("final_submission1.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (w*2,h*2))

while True:
    r1,f1 = v1.read()
    r2,f2 = v2.read()
    r3,f3 = v3.read()
    r4,f4 = v4.read()
    if not r1: break

    top = np.hstack((f1,f2))
    bottom = np.hstack((f3,f4))
    final = np.vstack((top,bottom))

    out.write(final)

out.release()
