import cv2
import os

train_image_folder = "../train/"
clean_images_folder = "../train_cleaned/"

def prepare_data_set(folder_dirty,folder_clean, size = 28):
    image_files = os.listdir(folder_dirty)
    dirty_images = []
    clean_images = []
    for i in range(0,len(image_files)):
        curr_image = cv2.imread(folder_dirty + image_files[i])
        curr_clean = cv2.imread(folder_clean + image_files[i])
        curr_shape = curr_image.shape
        num_w = curr_shape[0]//size
        num_h = curr_shape[0]//size
        for x in range(num_w):
            for y in range(num_h):
                dirty_images.append(curr_image[x*size:x*size+28,y*size:y*size+28,0])
                clean_images.append(curr_clean[x*size:x*size+28,y*size:y*size+28,0])
    return {"dirty":dirty_images,"clean":clean_images}

    



