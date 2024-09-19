import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Read the CSV file
df = pd.read_csv('hand_landmarks.csv', header=None)
landmark_data = df.iloc[:, 1:].values

def update(frame):
    ax.cla()  # Clear the previous frame
    landmarks = landmark_data[frame]
    
    # Extract x, y, z coordinates
    x = landmarks[0::3]
    y = landmarks[1::3]
    z = landmarks[2::3]
    
    # Plot the landmarks
    ax.scatter(x, y, z, c='r', marker='o')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {frame}')

# Plot the landmarks
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the animation
ani = FuncAnimation(fig, update, frames=len(landmark_data), interval=20)

plt.show()