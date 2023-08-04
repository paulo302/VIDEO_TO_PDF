import argparse
import cv2
from matplotlib import pyplot as plt
import time
import preprocessing
import utils
import math
import numpy as np

ratio = 10
start = time.time()
frame_rate = 5

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="  Put the path of video")
args = vars(ap.parse_args())

if not args['path']:
    print('Invalid argument!!')
    exit(1)

#Video compression to reduce processing time
video_fname = args['path']
compressed_video = preprocessing.compression(video_fname, frame_rate)

# Open the video
cap = cv2.VideoCapture(compressed_video)

# Get the total number of frames of the cap object
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Checks if the video opened correctly
if not cap.isOpened():
    print('ERROR: Failed to open the video')
    exit(1)

# Calculate threshold and return a list
percentage = 0.97
threshold, list_optical_flow = utils.Calc_threshold(ratio, percentage, cap)

# Output from frame selection
cap = cv2.VideoCapture(compressed_video)
utils.Frames_selection(list_optical_flow, threshold, cap)
cap.release()

# Print processing time (compression + processing code).
stop = time.time()
print("Runtime:", stop - start, "seconds")
print(total_frames * frame_rate)
print(threshold)
"""
# Make the histogram
results = list(map(math.log, list_optical_flow))
plt.hist(results, bins=1000)

# Label configuration
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram')

# Display the histogram
plt.show()
"""