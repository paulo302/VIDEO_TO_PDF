import numpy as np
import cv2
import preprocessing

"""Metric function takes two numpy matrices of frame resolution size calculates 
Euclidean norm of its elements"""

def Metric(flow):
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    matrice_norm = np.sqrt(fx*fx + fy*fy)
    return np.median(matrice_norm)

def Calc_threshold(ratio, percentage, cap):
    # Initialize the last frame
    ret, previous_frame = cap.read()

    # Call preprocessing function
    previous_frame_gray = preprocessing.preprocessing(previous_frame, ratio)

    # Creates an empty list to receive the magnitude of the optical flux vectors
    list_optical_flow = []

    # Process the frames from video
    while True:
        # Read the current frame
        ret, frame = cap.read()
        # If don't have any frame break the loop
        if not ret:
            break

        frame_gray = preprocessing.preprocessing(frame, ratio)

        # Calculate optical flow by farneback
        flow = cv2.calcOpticalFlowFarneback(prev=previous_frame_gray, next=frame_gray, flow=None,
                                            pyr_scale=0.5, levels=3, winsize=5, iterations=5,
                                            poly_n=5, poly_sigma=1.1, flags=0)

        # Extracts the median magnitude of the optical flow
        meadian_value = Metric(flow)

        list_optical_flow.append(meadian_value)

        # Wait for the "q" key to exit the process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Frames recursion
        previous_frame_gray = frame_gray

    sorted_list = sorted(list_optical_flow)

    # Calculates the index corresponding to the percentage of data.
    index_percentage = round(len(list_optical_flow) * percentage)
    # Accesses the approximate value corresponding to a percentage
    threshold = sorted_list[index_percentage]

    cap.release()

    return threshold, list_optical_flow

def Frames_selection(list, threshold, cap):
    count = 1
    i = 0
    while True:
        # Read the current frame
        ret, frame = cap.read()
        # If don't have any frame break the loop
        if not ret:
            break
        if i < len(list) and list[i] > threshold:
            output_path = '/home/paulo/Desktop/Python/Processamento_de_imagens/Projeto_Final/Output_folder/imagem' + \
                        str(count) + '.jpg'
            cv2.imwrite(output_path, frame)
            count += 1
        i += 1
    cap.release()