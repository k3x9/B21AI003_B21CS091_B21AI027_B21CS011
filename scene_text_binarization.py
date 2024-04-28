import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from utils import *

def method(FILE_NAME = None, FORMAT = None):
    # BLOCK-1: Text Enchanment
    if FILE_NAME is None or FORMAT is None:
        print("Please Input File Name (Example: image_name)")     
        FILE_NAME = input("FILE_NAME: ")
        print("Please Input File Format (Example: .png)") 
        FORMAT = input("FORMAT: ")

    text_enhancement(FILE_NAME, FORMAT)
    FILE_NAME = 'output_images/' + FILE_NAME + '/' + FILE_NAME

    # BLOCK-2: Text Binarization
    print("Binarization Started")
    file_suffixs = ['', '_Cei', '_CeiBin', '_EdgeBin', '_LDI', '_r', '_TLI', '_TLI_erosion']
    file_paths = []
    for suffix in file_suffixs:
        file_paths.append(FILE_NAME + suffix + FORMAT)

    original_image = cv2.imread(file_paths[0])
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    frames = []
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    for path in file_paths:
        if not os.path.exists(path):
            print('File not found: ', path)
            print('Please try again.')
            exit()
        
        frame = cv2.imread(path)
        frame = cv2.resize(frame, (320, 320))
        blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (123.68, 116.78, 103.94), True, False)

        output_layers = []
        output_layers.append("feature_fusion/Conv_7/Sigmoid")
        output_layers.append("feature_fusion/concat_3")

        net.setInput(blob)
        output = net.forward(output_layers)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores = output[0]
        geometry = output[1]
        
        frames.append((frame, scores, geometry))

        print(f"File {path} has been processed.")

    confThreshold = 0.5
    nmsThreshold = 0.3

    indices_arr = []
    boxes_arr = []

    for frame, scores, geometry in frames:
        [boxes, confidences] = decode_boxes(scores, geometry, confThreshold)
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
        indices_arr.append(indices)
        boxes_arr.append(boxes)

    print("Boxes and indices have been detected.")

    binarized_images = []
    fig, axs = plt.subplots(3,3,figsize = (10, 10))

    axs[0][0].imshow(original_image, cmap='gray')
    axs[0][0].axis('off')
    for i, frame in enumerate(frames[1:]):

        frame_, _, _ = frame
        binarize_image = mask_image(frame_, boxes_arr[i], indices_arr[i])
        binarized_images.append(binarize_image)

        # Save binarized image
        file_ = file_paths[i+1].split('/')[-1].split('.')[0] + '_binarized' + FORMAT
        image_file_path = '/'.join(file_paths[i+1].split('/')[:-1]) + '/' + file_
        cv2.imwrite(image_file_path, binarize_image * 255)

        i += 1
        axs[i//3][i%3].imshow(binarize_image, cmap='gray', vmin=0, vmax=1)
        axs[i//3][i%3].axis('off')
        rect = patches.Rectangle((0, 0), 320, 320, linewidth=2, edgecolor='black', facecolor='none')
        axs[i//3][i%3].add_patch(rect)

        print(f'File {image_file_path} has been saved.')
    axs[2][2].axis('off')

    file_name = file_paths[0].split('/')[-1].split('.')[0]
    plt.tight_layout()
    plt.savefig(f'output_images/{file_name}/{file_name}_collage{FORMAT}')
    print("Binarization Completed. Image saved")
    print('--------------------------------------')
    plt.show()

    return binarized_images

def get_binary_images(images):
    binarized_images = []
    for images in images:
        file_name, format = images.split('/')[-1].split('.')
        format = '.' + format
        bi = method(file_name, format)
        binarized_images.append(bi)
    return binarized_images


if __name__ == '__main__':
    temp = method()