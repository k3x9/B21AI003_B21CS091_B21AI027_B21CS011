    

# **Scene Text Binarization and Text Recognition**

The following code is used perform text binarization on the given scenes and later apply text recognition to recognize the text in the image.

## Specifications

* **scene_text_binarization.py** - Takes image as input and return text binarized images.
* **utils.py** - Contains the necessary utility functions used in `scene_text_binarization.py`
* **code.ipynb** - The following is being used to showcase the output images after appying binarization on the images.
* **frozen_east_text_detection.pb** - Pre-trained model used for text-detection.
* **Optical Character Recognition.ipynb** - After getting the text-binarized images, easy-ocr is being applied on the images to recognize the text. The file applies easy-ocr on the image and plots the output after text recognition.

## Compilation and Execution

* Place the image in `input_images` folder
* run `python scene_text_binarization.py`
* Output images will be saved in `output_images/<image_name>`
* The results can also be reviewed by running the cells of the `code.ipynb` and `Optical Character Recgognition.ipynb`.

## Example

Input Image:

![image](input_images/image3.jpeg)

After binarization:

![image](output_images/image3/image3_Cei_binarized.jpeg)
