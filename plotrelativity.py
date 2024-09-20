import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import sys

# Get the label from command line arguments if provided
label = sys.argv[1] if len(sys.argv) > 1 else None

# Define paths to dynamic and static data directories
dynamic_data_dir = 'data/dynamic'
static_data_dir = 'data/static'

# Function to read CSV files from a directory
def read_csv_files(directory, label=None):
    data = []
    files = os.listdir(directory)
    for file in files:
        if file.endswith('.csv'):
            file_label = os.path.splitext(file)[0]
            if label is None or file_label == label:
                file_path = os.path.join(directory, file)
                try:
                    df = pd.read_csv(file_path, header=None, on_bad_lines='skip')
                    data.append((file_label, df))
                except pd.errors.EmptyDataError:
                    print(f"Skipping empty file: {file_path}")
    return data

# Read dynamic and static data
dynamic_data = read_csv_files(dynamic_data_dir, label)
static_data = read_csv_files(static_data_dir, label)

# Number of landmarks (21 points with x, y, z coordinates)
num_landmarks = 21 * 3
feature_list_length = 91

# Define connections between landmarks
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Initialize a list to store the positions for tracing the path
path_positions = []

def update_dynamic(frame, data):
    global path_positions
    if frame == 0:
        path_positions = []  # Clear the path at the start of a new animation sequence
    
    ax1.cla()  # Clear the previous frame
    ax2.cla()
    ax3.cla()
    
    # Set background color to black
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    ax3.set_facecolor('black')
    
    # Calculate the start and end indices for the current frame
    start_idx = frame * feature_list_length
    end_idx = start_idx + num_landmarks
    
    # Extract x, y, z coordinates for landmarks
    landmark_frame = data.iloc[0, start_idx:end_idx].values
    x = landmark_frame[0::3]
    y = landmark_frame[1::3]
    z = landmark_frame[2::3]
    
    # Plot the landmarks
    ax1.scatter(x, y, z, c='white', marker='o')
    ax1.set_xlabel('X', color='white')
    ax1.set_ylabel('Y', color='white')
    ax1.set_zlabel('Z', color='white')
    ax1.set_title(f'Dynamic Landmarks - Frame {frame}', color='white')
    ax1.tick_params(colors='white')
    
    # Draw lines between the landmarks
    for connection in connections:
        start, end = connection
        ax1.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 'white')
    
    # Extract and plot the distances
    distance_frame = data.iloc[0, end_idx:end_idx + 12].values
    ax2.plot(distance_frame, 'wo-')
    ax2.set_xlabel('Distance Index', color='white')
    ax2.set_ylabel('Distance', color='white')
    ax2.set_title(f'Dynamic Distances - Frame {frame}', color='white')
    ax2.tick_params(colors='white')
    
    # Extract and plot the angles
    angle_frame = data.iloc[0, end_idx + 12:end_idx + 25].values
    ax3.plot(angle_frame, 'wo-')
    ax3.set_xlabel('Angle Index', color='white')
    ax3.set_ylabel('Angle (degrees)', color='white')
    ax3.set_title(f'Dynamic Angles - Frame {frame}', color='white')
    ax3.tick_params(colors='white')
    
    # Extract and plot the positions
    position_start_idx = end_idx + 25
    position_end_idx = position_start_idx + 3
    position_frame = data.iloc[0, position_start_idx:position_end_idx].values
    ax1.scatter(position_frame[0], position_frame[1], position_frame[2], c='red', marker='x', s=100)
    
    # Append the current position to the path_positions list
    path_positions.append(position_frame)
    
    # Plot the path taken by the hand landmarks
    if len(path_positions) > 1:
        path_positions_array = np.array(path_positions)
        ax1.plot(path_positions_array[:, 0], path_positions_array[:, 1], path_positions_array[:, 2], c='yellow')

def plot_static(data):
    ax1.cla()  # Clear the previous frame
    ax2.cla()
    ax3.cla()
    
    # Set background color to black
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    ax3.set_facecolor('black')
    
    # Extract x, y, z coordinates for landmarks
    landmark_frame = data.iloc[0, :num_landmarks].values
    x = landmark_frame[0::3]
    y = landmark_frame[1::3]
    z = landmark_frame[2::3]
    
    # Plot the landmarks
    ax1.scatter(x, y, z, c='white', marker='o')
    ax1.set_xlabel('X', color='white')
    ax1.set_ylabel('Y', color='white')
    ax1.set_zlabel('Z', color='white')
    ax1.set_title('Static Landmarks', color='white')
    ax1.tick_params(colors='white')
    
    # Draw lines between the landmarks
    for connection in connections:
        start, end = connection
        ax1.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 'white')
    
    # Plot the distances
    distance_frame = data.iloc[0, num_landmarks:num_landmarks + 12].values
    ax2.plot(distance_frame, 'wo-')
    ax2.set_xlabel('Distance Index', color='white')
    ax2.set_ylabel('Distance', color='white')
    ax2.set_title('Static Distances', color='white')
    ax2.tick_params(colors='white')
    
    # Plot the angles
    angle_frame = data.iloc[0, num_landmarks + 12:num_landmarks + 25].values
    ax3.plot(angle_frame, 'wo-')
    ax3.set_xlabel('Angle Index', color='white')
    ax3.set_ylabel('Angle (degrees)', color='white')
    ax3.set_title('Static Angles', color='white')
    ax3.tick_params(colors='white')
    
    # Extract and plot the positions
    position_frame = data.iloc[0, num_landmarks + 25:num_landmarks + 28].values
    ax1.scatter(position_frame[0], position_frame[1], position_frame[2], c='red', marker='x', s=100)

# Create the figure and subplots
fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# Set figure background color to black
fig.patch.set_facecolor('black')

# Plot dynamic data if available
if dynamic_data:
    for label, data in dynamic_data:
        print(f"Plotting dynamic data for label: {label}")
        print(data.shape)
        num_frames = data.shape[1] // feature_list_length
        ani = FuncAnimation(fig, update_dynamic, frames=num_frames, fargs=(data,), interval=100)
else:
    # Plot static data if no dynamic data is available
    for label, data in static_data:
        plot_static(data)

plt.show()