import cv2
import mediapipe as mp
import numpy as np
import argparse
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--label_index", type=int, required=True)
    args = parser.parse_args()
    return args

def logging_csv(label_index, landmark_list):
    csv_path = "hand_landmarks.csv"
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label_index] + landmark_list)

def main():
    args = get_args()
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(image, 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
                for hand_landmarks, hand_world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                    landmark_list = []
                    for landmark in hand_landmarks.landmark:
                        landmark_list.extend([landmark.x, landmark.y, landmark.z])
                    logging_csv(args.label_index, landmark_list)

                    for landmark, world_landmark in zip(hand_landmarks.landmark, hand_world_landmarks.landmark):
                        z_normalized = int(np.interp(world_landmark.z, [-0.1, 0.1], [255, 0]))
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(image, (x, y), 5, (z_normalized, z_normalized, z_normalized), -1)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == "__main__":
    main()