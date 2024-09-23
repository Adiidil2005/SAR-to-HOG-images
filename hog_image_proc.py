from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
from scipy.ndimage import median_filter
from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob

def HOG_images(images):
    img = cv2.imread(images)
    plt.axis("off")
    
    denoised_image = median_filter(img, size=3)
    #resizing image
    resized_img = resize(denoised_image, (128*4, 64*4))
    plt.axis("off")
    
    print(resized_img.shape)

    #creating hog features
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    
    
    #emboss kernal for image sharpening
    kernel = np.array([[-2, -1, 0], 
                       [-1, 1, 1], 
                       [0, 1, 2]]) 
    # Sharpen the image 
    sharpened_image = cv2.filter2D(hog_image, -1, kernel)
    return sharpened_image
    



def image_process():
    path = "SAR_target_detection/SAR_target_images/15_DEG/BTR_60/*.jpg"  # Match all .jpg files in the directory
    output_folder = "SAR_target_detection/SAR_target_images/15_DEG/HOG_BTR_60/"
    
    os.makedirs(output_folder, exist_ok=True)
    
    image_num = 0  

    # Iterate through all image files in the specified folder
    for file in glob.glob(path):
        print(f"Processing file: {file}")
        
        # Generate the HOG image
        HOG_image = img_as_ubyte(HOG_images(file) ) # Pass the full image path and convert it to 8 bit numbrs
        
        if HOG_image is not None:
            # Save the processed HOG image
            output_path = os.path.join(output_folder, f"BTR60_{image_num}.jpg")
            cv2.imwrite(output_path, HOG_image)
            print(f"Saved HOG image to: {output_path}")
            image_num += 1  
    
    print(f"The number of processed images is {image_num}")

    
image_process()




