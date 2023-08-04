import cv2
import subprocess
import numpy as np

"""Function that filters out salt and pepper noise via Median Filter contrast enhancement via Gaussian filter
and resize the frame via resize by percentage rate in relation to the image size"""

def preprocessing(img, ratio=100):
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.medianBlur(img, 5)

    h, w = img.shape
    img = cv2.resize(img, (int(ratio * h / 100), int(ratio * w / 100)))
    return img

# compression by subprocess library, ffmpeg to reduce frame rate. 
def compression(path, frame_rate=5):
    input_file = path
    output_file = input_file.split(".mp4")[0] + '_reduzido.mp4'
    command = f'ffmpeg -i {input_file} -r  {frame_rate} {output_file}'
    subprocess.call(command, shell=True)

    return output_file
