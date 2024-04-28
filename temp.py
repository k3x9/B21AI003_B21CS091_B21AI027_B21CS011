import os
FILE_NAME = '7'
FORMAT = '.jpg'
FILE_NAME_OUT_FOLDER_NAME = 'output_images/' + FILE_NAME
FILE_NAME, temp = 'input_images/' + FILE_NAME, FILE_NAME
os.makedirs(FILE_NAME_OUT_FOLDER_NAME, exist_ok=True)

import shutil
shutil.copy(FILE_NAME + FORMAT, FILE_NAME_OUT_FOLDER_NAME + '/' + temp + FORMAT)