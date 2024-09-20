import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# Get the label from command line arguments if provided
label = sys.argv[1] if len(sys.argv) > 1 else None

# Read the CSV file
df = pd.read_csv('hand_landmarks.csv', header=None)

# Filter by label if specified
if label:
    label = label.capitalize()
    df = df[df[0] == label]

landmark_data = df.iloc[:, 1:].values
frame_labels = df.iloc[:, 0].values

# Number of landmarks (21 points with x, y, z coordinates)
num_landmarks = 21 * 3

# Extract landmark coordinates, distances, and angles
landmarks = landmark_data[:, :num_landmarks]
distances = landmark_data[:, num_landmarks:num_landmarks + 12]
angles = landmark_data[:, num_landmarks + 12:]

# Define connections between landmarks
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def update(frame):
    ax1.cla()  # Clear the previous frame
    ax2.cla()
    ax3.cla()
    
    # Set background color to black
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    ax3.set_facecolor('black')
    
    # Extract x, y, z coordinates for landmarks
    landmark_frame = landmarks[frame]
    x = landmark_frame[0::3]
    y = landmark_frame[1::3]
    z = landmark_frame[2::3]
    
    # Plot the landmarks
    ax1.scatter(x, y, z, c='white', marker='o')
    ax1.set_xlabel('X', color='white')
    ax1.set_ylabel('Y', color='white')
    ax1.set_zlabel('Z', color='white')
    ax1.set_title(f'Landmarks - Frame {frame} - Label {frame_labels[frame]}', color='white')
    ax1.tick_params(colors='white')
    
    # Draw lines between the landmarks
    for connection in connections:
        start, end = connection
        ax1.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 'white')
    
    # Plot the distances
    distance_frame = distances[frame]
    ax2.plot(distance_frame, 'wo-')
    ax2.set_xlabel('Distance Index', color='white')
    ax2.set_ylabel('Distance', color='white')
    ax2.set_title(f'Distances - Frame {frame} - Label {frame_labels[frame]}', color='white')
    ax2.tick_params(colors='white')
    
    # Plot the angles
    angle_frame = angles[frame]
    ax3.plot(angle_frame, 'wo-')
    ax3.set_xlabel('Angle Index', color='white')
    ax3.set_ylabel('Angle (degrees)', color='white')
    ax3.set_title(f'Angles - Frame {frame} - Label {frame_labels[frame]}', color='white')
    ax3.tick_params(colors='white')

# Create the figure and subplots
fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# Set figure background color to black
fig.patch.set_facecolor('black')

# Create the animation
ani = FuncAnimation(fig, update, frames=len(landmark_data), interval=100)

plt.show()