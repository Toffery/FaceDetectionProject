import time
import mediapipe as mp
import cv2


class FaceDetector():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mp_face_detect = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detect = self.mp_face_detect.FaceDetection()

    def find_faces(self, img, draw=True, draw_detection=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detect.process(img_rgb)
        bound_boxes = []

        if draw:
            if self.results.detections:
                for id, detection in enumerate(self.results.detections):
                    if draw_detection:
                        self.mp_draw.draw_detection(img, detection)
                    bounding_box_class = detection.location_data.relative_bounding_box
                    height, width, channels = img.shape
                    bounding_box = int(bounding_box_class.xmin * width), int(bounding_box_class.ymin * height), \
                                   int(bounding_box_class.width * width), int(bounding_box_class.height * height)
                    bound_boxes.append([id, bounding_box, detection.score])
                    cv2.rectangle(img, bounding_box, (0, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bounding_box[0], bounding_box[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return img, bound_boxes


fps_counter = []
def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxes = detector.find_faces(img, draw=True, draw_detection=False)

        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        fps_counter.append(fps)
        prev_time = cur_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow('Dolboeb', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
