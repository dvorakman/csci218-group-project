import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def check_proximity(hand1, hand2):
    hand1_points = [(lm.x, lm.y, lm.z) for lm in hand1]
    hand2_points = [(lm.x, lm.y, lm.z) for lm in hand2]
    
    tree1 = KDTree(hand1_points)
    tree2 = KDTree(hand2_points)
    
    for point in hand1_points:
        if tree2.query_ball_point(point, r=0.1):
            return True
    return False

def plot_hand_landmarks(ax, hand_landmarks_list):
    ax.clear()
    for hand_landmarks in hand_landmarks_list:
        x = [lm.x for lm in hand_landmarks.landmark]
        y = [lm.y for lm in hand_landmarks.landmark]
        z = [lm.z for lm in hand_landmarks.landmark]
        ax.plot(x, y, z, 'o-')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.draw()
    plt.pause(0.001)

def main():
    cap = cv.VideoCapture(0)
    hands = mp_hands.Hands()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    plt.show()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        debug_image = frame.copy()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks_list = [hand_landmarks for hand_landmarks in results.multi_hand_landmarks]
            if len(hand_landmarks_list) == 2:
                if check_proximity(hand_landmarks_list[0].landmark, hand_landmarks_list[1].landmark):
                    print("Hands are in proximity")
                    for hand_landmarks in hand_landmarks_list:
                        normalized_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        print("Normalized 3D coordinates:", normalized_landmarks)
                    plot_hand_landmarks(ax, hand_landmarks_list)

            for hand_landmarks in hand_landmarks_list:
                mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv.imshow("Hand Gesture Recognition", debug_image)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    hands.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()