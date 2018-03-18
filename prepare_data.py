import cv2
import os

train_image_folder = "../train/"
train_clean_folder = "../train_cleaned/"

def prepare_training_set(size=28):
    train_image_files = os.listdir(train_image_folder)
    train_images = []
    train_clean = []
    for i in range(0,len(train_image_files)):
        curr_image = cv2.imread(train_image_folder + train_image_files[i])
        curr_clean = cv2.imread(train_clean_folder + train_image_files[i])
        curr_shape = curr_image.shape
        num_w = curr_shape[0]//size
        num_h = curr_shape[0]//size
        for x in range(num_w):
            for y in range(num_h):
                train_images.append(curr_image[x*size:x*size+28,y*size:y*size+28,0])
                train_clean.append(curr_clean[x*size:x*size+28,y*size:y*size+28,0])
    return {"dirty":train_images,"clean":train_clean}



