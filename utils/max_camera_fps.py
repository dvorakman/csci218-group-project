import cv2

def get_max_fps():
    cap = cv2.VideoCapture(0)
    max_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return max_fps

max_fps = get_max_fps()
print("Maximum FPS: ", max_fps)