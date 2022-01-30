import time
import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
prev_time = 0

mp_face_detect = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detect = mp_face_detect.FaceDetection()


while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detect.process(img_rgb)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(img, detection)
            bounding_box_class = detection.location_data.relative_bounding_box
            height, width, channels = img.shape
            bounding_box = int(bounding_box_class.xmin * width), int(bounding_box_class.ymin * height), \
                            int(bounding_box_class.width * width), int(bounding_box_class.height * height)
            cv2.rectangle(img, bounding_box, (0, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%',
                        (bounding_box[0], bounding_box[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow('Window', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
