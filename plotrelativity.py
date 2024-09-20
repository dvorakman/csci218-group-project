import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Read the CSV file
df = pd.read_csv('hand_landmarks.csv', header=None)
landmark_data = df.iloc[:, 1:].values

# Number of landmarks (21 points with x, y, z coordinates)
num_landmarks = 21 * 3

# Extract landmark coordinates, distances, and angles
landmarks = landmark_data[:, :num_landmarks]
distances = landmark_data[:, num_landmarks:num_landmarks + 12]
angles = landmark_data[:, num_landmarks + 12:]

def update(frame):
    ax1.cla()  # Clear the previous frame
    ax2.cla()
    ax3.cla()
    
    # Extract x, y, z coordinates for landmarks
    landmark_frame = landmarks[frame]
    x = landmark_frame[0::3]
    y = landmark_frame[1::3]
    z = landmark_frame[2::3]
    
    # Plot the landmarks
    ax1.scatter(x, y, z, c='r', marker='o')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Landmarks - Frame {frame}')
    
    # Plot the distances
    distance_frame = distances[frame]
    ax2.plot(distance_frame, 'bo-')
    ax2.set_xlabel('Distance Index')
    ax2.set_ylabel('Distance')
    ax2.set_title(f'Distances - Frame {frame}')
    
    # Plot the angles
    angle_frame = angles[frame]
    ax3.plot(angle_frame, 'go-')
    ax3.set_xlabel('Angle Index')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title(f'Angles - Frame {frame}')

# Create the figure and subplots
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# Create the animation
ani = FuncAnimation(fig, update, frames=len(landmark_data), interval=100)

plt.show()