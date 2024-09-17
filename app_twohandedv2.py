import csv
import copy
import argparse
import itertools
from collections import Counter, deque


import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
from model import KeyPointClassifier
import time

# Constants
STABILITY_THRESHOLD = 15  # Number of consecutive frames to confirm gesture
SPACE_GESTURE_ID = None # Will be set after loading labels


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=1000)
    parser.add_argument("--height", help="cap height", type=int, default=1000)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=float,  # Changed to float for consistency
        default=0.5,
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['data_collection', 'recognition'],
        default='recognition',
        help="Mode of operation: 'data_collection' to collect data, 'recognition' to recognize gestures."
    )
    # Removed 'label_index' as we'll handle labels via buffer
    # parser.add_argument('--label_index', type=int, help='Label index for continuous data collection', default=None)

    args = parser.parse_args()

    return args


def main():
    global SPACE_GESTURE_ID  # To modify the global variable
    COOLDOWN_PERIOD = 5.0

    args = get_args()

    # New Buffers for Sentence Construction
    sentence_buffer = []  # List to store the sentence
    last_appended_gesture = None  # To track the last appended gesture
    last_append_time = 0  # Initialize last append time

    if args.mode == 'data_collection':
        continuous_data_collection(args)
        return  # Exit after data collection

    # Initialization for Recognition Mode
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    with open(
        "model/keypoint_classifier/keypoint_testing_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
        print(keypoint_classifier_labels)

    # Set SPACE_GESTURE_ID
    try:
        SPACE_GESTURE_ID = keypoint_classifier_labels.index('[space]')
    except ValueError:
        print("Error: 'space' label not found in keypoint_classifier_labels.")
        print("Please ensure that 'space' is a label in 'keypoint_testing_label.csv'.")
        exit(1)

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Initialize Buffers
    history_length = 16
    point_history = deque(maxlen=history_length)
    gesture_classification_history = deque(maxlen=history_length)

    mode = 0

    while args.mode != 'data_collection':
        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            hand_landmarks_list = []
            handedness_list = []

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                hand_landmarks_list.append((landmark_list, handedness.classification[0].label))
                handedness_list.append(handedness)

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

            hand_landmarks_list.sort(key=lambda x: x[1])

            # Combine landmarks of both hands
            if len(hand_landmarks_list) == 1:
                # Pad single-handed landmarks to match two-handed format
                padded_landmarks = pad_single_hand_landmarks(hand_landmarks_list[0][0], hand_landmarks_list[0][1])
                combined_landmarks = padded_landmarks
            elif len(hand_landmarks_list) == 2:
                # Combine landmarks of both hands
                combined_landmarks = hand_landmarks_list[0][0] + hand_landmarks_list[1][0]
            else:
                combined_landmarks = []

            if combined_landmarks:
                pre_processed_landmark_list = pre_process_landmark(combined_landmarks)
                current_label = 'Unknown'  # Default label

                logging_csv(current_label, 1, pre_processed_landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                gesture_classification_history.append(hand_sign_id)
                most_common_fg_id, count = Counter(gesture_classification_history).most_common(1)[0]

                # Check for stability
                if count >= STABILITY_THRESHOLD:
                    current_time = time.time()  # Get the current time
                    if most_common_fg_id != last_appended_gesture or (current_time - last_append_time >= COOLDOWN_PERIOD):
                        label = keypoint_classifier_labels[most_common_fg_id]
                        if most_common_fg_id == SPACE_GESTURE_ID:
                            sentence_buffer.append(" ")
                        elif label != 'Unknown':
                            sentence_buffer.append(label)
                        last_appended_gesture = most_common_fg_id
                        last_append_time = current_time

                # Combine landmarks of both hands
                combined_brect = calc_combined_bounding_rect(hand_landmarks_list)
                debug_image = draw_combined_bounding_rect(debug_image, combined_brect)
                debug_image = draw_info_text(
                    debug_image,
                    combined_brect,
                    handedness_list[0],  # Assuming the first hand's handedness for display
                    keypoint_classifier_labels[most_common_fg_id],
                    buffer_letters=''.join(sentence_buffer)  # Display the sentence buffer
                )

                # Update Point History (e.g., tip of the index finger)
                index_finger_tip = combined_landmarks[8]  # Index finger tip landmark
                point_history.append(index_finger_tip)

                # Optionally, you can visualize point history
                debug_image = draw_point_history(debug_image, point_history)

        # Update the display with the sentence buffer
        debug_image = draw_info(debug_image, fps, mode, ''.join(sentence_buffer))
        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


def continuous_data_collection(args):
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # label_index = args.label_index  # Removed since we are using buffer

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Initialize Buffers for Data Collection
    history_length = 16
    point_history = deque(maxlen=history_length)
    gesture_classification_history = deque(maxlen=history_length)
    # letter_buffer is used to store labels based on keypress, but in data collection, you might want to label based on mode
    letter_buffer = deque(maxlen=10)  # Buffer to store up to 10 recent letters

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode, letter = select_mode(key, 1)  # Force mode=1 for data collection
        if letter:
            letter_buffer.append(letter)  # Add detected letter to buffer

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            hand_landmarks_list = []
            handedness_list = []

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                hand_landmarks_list.append((landmark_list, handedness.classification[0].label))
                handedness_list.append(handedness)

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

            hand_landmarks_list.sort(key=lambda x: x[1])

            # Combine landmarks of both hands
            if len(hand_landmarks_list) == 1:
                # Pad single-handed landmarks to match two-handed format
                padded_landmarks = pad_single_hand_landmarks(hand_landmarks_list[0][0], hand_landmarks_list[0][1])
                combined_landmarks = padded_landmarks
            elif len(hand_landmarks_list) == 2:
                # Combine landmarks of both hands
                combined_landmarks = hand_landmarks_list[0][0] + hand_landmarks_list[1][0]
            else:
                combined_landmarks = []

            if combined_landmarks:
                pre_processed_landmark_list = pre_process_landmark(combined_landmarks)
                # Use the latest letter from the buffer as the label
                if letter_buffer:
                    current_label = letter_buffer[-1]
                else:
                    current_label = 'Unknown'  # Default label if buffer is empty

                logging_csv(current_label, 1, pre_processed_landmark_list)
                # Assuming you want to classify gestures here as well
                # hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Update Gesture Classification History
                # gesture_classification_history.append(hand_sign_id)
                # most_common_fg_id = Counter(gesture_classification_history).most_common(1)[0][0]

                combined_brect = calc_combined_bounding_rect(hand_landmarks_list)
                debug_image = draw_combined_bounding_rect(debug_image, combined_brect)
                debug_image = draw_info_text(
                    debug_image,
                    combined_brect,
                    handedness_list[0],  # Assuming the first hand's handedness for display
                    current_label,  # Use current_label directly
                    buffer_letters=''.join(letter_buffer)  # Pass buffered letters for display
                )

                # Update Point History (e.g., tip of the index finger)
                index_finger_tip = combined_landmarks[8]  # Index finger tip landmark
                point_history.append(index_finger_tip)

                # Optionally, you can visualize point history
                debug_image = draw_point_history(debug_image, point_history)

        debug_image = draw_info(debug_image, fps, mode, ''.join(letter_buffer))  # Display buffered letters
        cv.imshow('Continuous Data Collection', debug_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def draw_combined_bounding_rect(image, brect):
    cv.rectangle(image, (int(brect[0]), int(brect[1])), (int(brect[2]), int(brect[3])), (0, 255, 0), 2)
    return image


def pad_single_hand_landmarks(landmark_list, handedness):
    # Assuming each hand has 21 joints, pad to 42 joints for two hands
    if handedness == 'Left':
        # Pad the right side
        return landmark_list + [[0, 0]] * 21
    elif handedness == 'Right':
        # Pad the left side
        return [[0, 0]] * 21 + landmark_list
    else:
        raise ValueError("Handedness must be 'Left' or 'Right'")


def select_mode(key, current_mode):
    """
    Modify select_mode to handle letter buffering.

    Returns:
        number: The numerical representation if a number key is pressed, else -1.
        mode: Updated mode.
        letter: The detected letter if a letter key is pressed, else None.
    """
    number = -1
    letter = None

    # Detect number keys '0' - '9'
    if 48 <= key <= 57:  # ASCII codes for '0' - '9'
        number = key - 48

    # Detect lowercase letters 'a' - 'z'
    if 97 <= key <= 122:  # ASCII codes for 'a' - 'z'
        letter = chr(key)

    # Mode switching
    if key == ord('\\'):  # Backslash to toggle off keypoint mode
        current_mode = 0
    if key == ord(' '):  # Spacebar to toggle on keypoint logging mode
        current_mode = 1
    # if key == ord('h'):  # 'h' to switch to mode 2
    #     current_mode = 2

    return number, current_mode, letter


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_combined_bounding_rect(hand_landmarks_list):
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')

    for landmarks, _ in hand_landmarks_list:
        for x, y in landmarks:
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y

    return [x_min, y_min, x_max, y_max]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        if landmark_point[0] == 0 and landmark_point[1] == 0:
            continue

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(label, mode, landmark_list, point_history_list=None):
    """
    Log the gesture data with the associated label.

    Args:
        label (str): The label associated with the gesture (e.g., a letter).
        mode (int): Mode of operation (e.g., 1 for data collection).
        landmark_list (list): Preprocessed landmark data.
        point_history_list (list, optional): History of points. Defaults to None.
    """
    if mode == 0:
        pass
    if mode == 1 and isinstance(label, str):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label, *landmark_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Define connections for drawing based on MediaPipe's hand landmarks
        connections = [
            (2, 3), (3, 4),        # Thumb
            (5, 6), (6, 7), (7, 8),  # Index
            (9, 10), (10, 11), (11, 12),  # Middle
            (13, 14), (14, 15), (15, 16),  # Ring
            (17, 18), (18, 19), (19, 20),  # Little
            (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)  # Palm
        ]

        for start, end in connections:
            cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (255, 255, 255), 2)

        # Draw keypoints
        for index, landmark in enumerate(landmark_point):
            if index in [4, 8, 12, 16, 20]:  # Fingertips
                radius = 8
                color = (255, 255, 255)
            else:
                radius = 5
                color = (255, 255, 255)
            cv.circle(image, (landmark[0], landmark[1]), radius, color, -1)
            cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text, buffer_letters=""):
    # Draw a filled rectangle for text background
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    # Prepare info text
    info_text = handedness.classification[0].label
    if hand_sign_text:
        info_text = f"{info_text}: {hand_sign_text}"

    # Put the gesture info text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    # Optionally, display buffered letters (sentence)
    if buffer_letters:
        cv.putText(
            image,
            f"Sentence: {buffer_letters}",
            (10, brect[3] + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

    return image


def draw_info(image, fps, mode, buffered_letters):
    # Display FPS
    cv.putText(
        image,
        f"FPS: {fps}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        f"FPS: {fps}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    # Display Mode and Buffered Letters
    mode_string = ["Idle", "Logging Key Point"]
    if mode == 1:
        cv.putText(
            image,
            f"MODE: {mode_string[mode - 1]}",
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if buffered_letters:
            cv.putText(
                image,
                f"Sentence: {buffered_letters}",
                (10, 120),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(
                image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2
            )
    return image


def collect_data(args):
    """
    This function is deprecated as we've integrated data collection into the main function
    using the 'data_collection' mode.
    """
    pass  # Implemented within main()


if __name__ == "__main__":
    main()