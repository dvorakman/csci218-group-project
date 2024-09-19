import argparse
import copy
import csv
import cv2 as cv
import itertools
import mediapipe as mp
import numpy as np
from model import KeyPointClassifier
from utils.cvfpscalc import CvFpsCalc

import asyncio
from scipy.spatial import KDTree

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)

    parser.add_argument('--mode', type=str, choices=['data_collection', 'recognition'], default='recognition')
    parser.add_argument('--label_index', type=int, default=None)

    args = parser.parse_args()

    if args.mode == 'data_collection' and args.label_index is None:
        parser.error("--label_index is required when mode is 'data_collection'")

    return args

def setup_capture(args):
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap

def setup_hands(args):
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

def load_labels(filepath):
    with open(filepath, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f)]

async def process_image(cap, hands):
    ret, image = cap.read()
    if not ret:
        return None, None
    image = cv.flip(image, 1)
    debug_image = image.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    return debug_image, results

def pad_single_hand_landmarks(landmark_list, handedness):
    if handedness == 'Left':
        return landmark_list + [[0, 0]] * 21, handedness

    elif handedness == 'Right':
        return [[0, 0]] * 21 + landmark_list, handedness
    
    elif handedness == 'Both':
        return landmark_list, handedness

    else:
        raise ValueError("Handedness must be 'Left' or 'Right' or 'Both'")

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        radius = 5 if index != 4 and index != 8 and index != 12 and index != 16 and index != 20 else 8
        cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image

def handle_landmarks(debug_image, results):
    hand_landmarks_list = []
    handedness_list = []
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
        hand_landmarks_list.append((landmark_list, handedness.classification[0].label))
        handedness_list.append(handedness)
        debug_image = draw_landmarks(debug_image, landmark_list)
    hand_landmarks_list.sort(key=lambda x: x[1])
    return hand_landmarks_list, handedness_list, debug_image

def pre_process_landmark(landmark_list):
    base_y = landmark_list[0][1]
    temp_landmark_list = [[x, y - base_y] for x, y in landmark_list]
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list))
    return [n / max_value for n in temp_landmark_list]

def logging_csv(number, landmark_list):
    csv_path = "model/keypoint_classifier/keypoint.csv"
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])

def calc_combined_bounding_rect(hand_landmarks_list):
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for landmarks, _ in hand_landmarks_list:
        for x, y in landmarks:
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)
    return x_min, y_min, x_max, y_max

def draw_bounding_rect(image, brect):
    cv.rectangle(image, (int(brect[0]), int(brect[1])), (int(brect[2]), int(brect[3])), (0, 255, 0), 2)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = str(handedness)
    if hand_sign_text:
        info_text += f": {hand_sign_text}"

    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
        cv.LINE_AA,
    )

    return image

def draw_info(image, fps):
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    return image

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

def bounding_rect(points):
    x_coordinates = [point[0] for point in points if point != [0, 0]]
    y_coordinates = [point[1] for point in points if point != [0, 0]]

    x_min = min(x_coordinates)
    y_min = min(y_coordinates)
    x_max = max(x_coordinates)
    y_max = max(y_coordinates)

    return [x_min, y_min, x_max, y_max]

def combine_landmarks(hand_landmarks_list, threshold=75):
    def are_hands_close(hand1, hand2, threshold):
        tree1 = KDTree(hand1)
        tree2 = KDTree(hand2)
        for point in hand1:
            if tree2.query_ball_point(point, threshold):
                return True
        for point in hand2:
            if tree1.query_ball_point(point, threshold):
                return True
        return False

    while len(hand_landmarks_list) > 1:
        combined = False
        for i in range(len(hand_landmarks_list)):
            for j in range(i + 1, len(hand_landmarks_list)):
                if are_hands_close(hand_landmarks_list[i][0], hand_landmarks_list[j][0], threshold):
                    combined_landmarks = hand_landmarks_list[i][0] + hand_landmarks_list[j][0]
                    hand_landmarks_list = [
                        hand for k, hand in enumerate(hand_landmarks_list) if k != i and k != j
                    ]
                    hand_landmarks_list.append((combined_landmarks, "Both"))
                    combined = True
                    break
            if combined:
                break
        if not combined:
            break
    
    return [pad_single_hand_landmarks(hand[0], hand[1]) for hand in hand_landmarks_list]

async def main():
    args = get_args()
    cap = setup_capture(args)
    hands = setup_hands(args)
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    keypoint_classifier = KeyPointClassifier() if args.mode != 'data_collection' else None
    keypoint_classifier_labels = load_labels("model/keypoint_classifier/keypoint_testing_label.csv")

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(1)

        if key == 27:
            break

        debug_image, results = await process_image(cap, hands)
        if debug_image is None:
            break

        if results.multi_hand_landmarks:
            hand_landmarks_list, handedness_list, debug_image = handle_landmarks(debug_image, results)
            hand_list = combine_landmarks(hand_landmarks_list)

            for hand in hand_list:
                brect = bounding_rect(hand[0])

                debug_image = draw_bounding_rect(debug_image, brect)
                pre_processed_landmark_list = pre_process_landmark(hand[0])
                if args.mode == 'data_collection':
                    logging_csv(args.label_index, pre_processed_landmark_list)

                else:
                    try:
                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        debug_image = draw_info_text(debug_image, brect, hand[1], keypoint_classifier_labels[hand_sign_id])
                    except ValueError as e:
                        print(f"dev note - idk how to fix this easily and cbf. \n\tOriginal Message: '{e}'")

        debug_image = draw_info(debug_image, fps)
        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())