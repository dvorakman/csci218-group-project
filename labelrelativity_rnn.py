import cv2
import mediapipe as mp
import numpy as np
import argparse
import h5py
import math
from collections import deque
import os
import tensorflow as tf

# Load the pre-trained models
cnn_model = tf.keras.models.load_model('hand_landmarks_cnn_model.keras')
rnn_model = tf.keras.models.load_model('relativitymatters_rnn_model.keras')

# Define the hand gesture labels
static_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
dynamic_labels = ['J', 'Z']
all_labels = static_labels + dynamic_labels

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    return args

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

def calculate_angle(point1, point2, point3):
    a = np.array([point1.x, point1.y, point1.z])
    b = np.array([point2.x, point2.y, point2.z])
    c = np.array([point3.x, point3.y, point3.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def moving_average(data, window_size):
    return np.mean(list(data)[-window_size:], axis=0)

def main():
    args = get_args()

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands

    # Buffer to store previous frames for smoothing
    buffer_size = 5
    landmark_buffer = deque(maxlen=buffer_size)
    distance_buffer = deque(maxlen=buffer_size)
    angle_buffer = deque(maxlen=buffer_size)

    # Buffer to store sequences for dynamic gesture detection
    sequence_length = 10 # Sliding window size
    sequence_buffer = deque(maxlen=sequence_length)

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
                    feature_list = []
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    landmark_buffer.append(landmarks)
                    
                    # Calculate distances between key points
                    distances = [
                        calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[4]),
                        calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[8]),
                        calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[12]),
                        calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[16]),
                        calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[20]),
                        calculate_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8]),
                        calculate_distance(hand_landmarks.landmark[8], hand_landmarks.landmark[12]),
                        calculate_distance(hand_landmarks.landmark[12], hand_landmarks.landmark[16]),
                        calculate_distance(hand_landmarks.landmark[16], hand_landmarks.landmark[20]),
                        calculate_distance(hand_landmarks.landmark[5], hand_landmarks.landmark[9]),
                        calculate_distance(hand_landmarks.landmark[9], hand_landmarks.landmark[13]),
                        calculate_distance(hand_landmarks.landmark[13], hand_landmarks.landmark[17])
                    ]
                    distance_buffer.append(distances)

                    # Calculate angles between joints
                    angles = [
                        calculate_angle(hand_landmarks.landmark[1], hand_landmarks.landmark[2], hand_landmarks.landmark[3]),
                        calculate_angle(hand_landmarks.landmark[2], hand_landmarks.landmark[3], hand_landmarks.landmark[4]),
                        calculate_angle(hand_landmarks.landmark[5], hand_landmarks.landmark[6], hand_landmarks.landmark[7]),
                        calculate_angle(hand_landmarks.landmark[6], hand_landmarks.landmark[7], hand_landmarks.landmark[8]),
                        calculate_angle(hand_landmarks.landmark[9], hand_landmarks.landmark[10], hand_landmarks.landmark[11]),
                        calculate_angle(hand_landmarks.landmark[10], hand_landmarks.landmark[11], hand_landmarks.landmark[12]),
                        calculate_angle(hand_landmarks.landmark[13], hand_landmarks.landmark[14], hand_landmarks.landmark[15]),
                        calculate_angle(hand_landmarks.landmark[14], hand_landmarks.landmark[15], hand_landmarks.landmark[16]),
                        calculate_angle(hand_landmarks.landmark[17], hand_landmarks.landmark[18], hand_landmarks.landmark[19]),
                        calculate_angle(hand_landmarks.landmark[18], hand_landmarks.landmark[19], hand_landmarks.landmark[20]),
                        calculate_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[5], hand_landmarks.landmark[9]),
                        calculate_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[9], hand_landmarks.landmark[13]),
                        calculate_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[13], hand_landmarks.landmark[17])
                    ]
                    angle_buffer.append(angles)

                    # Apply moving average to smooth the data
                    if len(landmark_buffer) == buffer_size:
                        smoothed_landmarks = moving_average(landmark_buffer, buffer_size)
                        smoothed_distances = moving_average(distance_buffer, buffer_size)
                        smoothed_angles = moving_average(angle_buffer, buffer_size)
                        
                        feature_list.extend(smoothed_landmarks)
                        feature_list.extend(smoothed_distances)
                        feature_list.extend(smoothed_angles)

                        sequence_buffer.append(feature_list)

                        # Predict using both models
                        if len(sequence_buffer) == sequence_length:
                            cnn_input = np.array(feature_list).reshape(1, -1, 1)
                            rnn_input = np.array(sequence_buffer).reshape(1, sequence_length, -1)

                            cnn_prediction = cnn_model.predict(cnn_input)
                            rnn_prediction = rnn_model.predict(rnn_input)

                            # Decision mechanism
                            cnn_confidence = np.max(cnn_prediction)
                            rnn_confidence = np.max(rnn_prediction)
                            
                            if rnn_confidence > 0.8:  # Adjust this threshold as needed
                                gesture = dynamic_labels[np.argmax(rnn_prediction)]
                            else:
                                gesture = static_labels[np.argmax(cnn_prediction)]

                            print(f"Predicted gesture: {gesture}")

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