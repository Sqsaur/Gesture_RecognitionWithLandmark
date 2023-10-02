import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import threading
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

timestamp = 0

model_path_gest = 'gesture_recognizer.task'
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path_hand = 'hand_landmarker.task'
BaseOptionsHandLandmarker = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningModeHandLandmarker = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:


class Mediapipe_BodyModule():
    def __init__(self):
        self.results = None

    def print_result_gest(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        print('gesture recognition result: {}'.format(result))
        self.results = result

    def print_result_hand(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        print('hand landmarker result: {}'.format(result))
        self.results = result

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image

    def main(self):

        options_gest = GestureRecognizerOptions(
            num_hands=2,
            base_options=BaseOptions(model_asset_path=model_path_gest),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result_gest)
        self.lock = threading.Lock()
        self.current_gestures = []
        options_hand = HandLandmarkerOptions(
            num_hands=2,
            base_options=BaseOptions(model_asset_path=model_path_hand),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result_hand)
        cap = cv2.VideoCapture(0)
        timestamp = 0
        with GestureRecognizer.create_from_options(options_gest) as recognizer, HandLandmarker.create_from_options(options_hand) as landmarker:
            # The recognizer is initialized. Use it here.
            while cap.isOpened():
                # Capture frame-by-frame
                ret, frame = cap.read()

                if not ret:
                    print("Ignoring empty frame")
                    break

                timestamp += 1
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=frame)
                # Send live image data to perform gesture recognition
                recognizer.recognize_async(mp_image, timestamp)
                landmarker.detect_async(mp_image, timestamp)
                if (not (self.results is None)):
                    annotated_image = self.draw_landmarks_on_image(
                        mp_image.numpy_view(), self.results)
                    self.put_gestures(annotated_image)
                    cv2.imshow('Show', annotated_image)
                    print("showing detected image")
                else:
                    cv2.imshow('Show', frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    print("Closing Camera Stream")

            # Release the VideoCapture and close all OpenCV windows
            cap.release()
            cv2.destroyAllWindows()

    def put_gestures(self, annotated_image):
        self.lock.acquire()
        gestures = self.current_gestures
        self.lock.release()
        y_pos = 50
        for hand_gesture_name in gestures:
            # show the prediction on the frame
            cv2.putText(annotated_image, hand_gesture_name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            y_pos += 50

    def print_result_gest(self, result, output_image, timestamp_ms):
        # print(f'gesture recognition result: {result}')
        self.lock.acquire()  # solves potential concurrency issues
        self.current_gestures = []
        if result is not None and any(result.gestures):
            print("Recognized gestures:")
            for single_hand_gesture_data in result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                print(gesture_name)
                self.current_gestures.append(gesture_name)
        self.lock.release()


if __name__ == "__main__":
    body_module = Mediapipe_BodyModule()
    body_module.main()
