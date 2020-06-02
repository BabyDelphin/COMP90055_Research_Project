# Data preprocess - Neal
import os
from os.path import join
import subprocess
import cv2
from PIL import Image
import sys

def detectFaces(img):
    """Method to detect the face from the picture"""
    # img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    result = []
    size = 0
    for (x, y, width, height) in faces:
        if width * height > size:
            if len(result) == 0:
                result.append((x, y, x + width, y + height))
                size = width * height
            else:
                result = result[:-1]
                result.append((x, y, x + width, y + height))
                size = width * height
    return result

def saveFaces(img, video_num, folder_name):
    """Method to crop the detected face from the picture"""
    img_size = 299
    faces = detectFaces(img)
    full_path = folder_name
    os.makedirs(full_path, exist_ok=True)

    for (x1, y1, x2, y2) in faces:
        video_name = video_num
        file_name = video_name

        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        img.crop((x1, y1, x2, y2)).resize((img_size, img_size), Image.LANCZOS).save(
            full_path + "/" + file_name)


if __name__ == '__main__':

    # add the terminal input into the arguments list
    arguments = sys.argv
    folder = arguments[1]
    all_video = sorted(os.listdir(folder))
    outputFolder = "faces"
    for i in range(len(all_video)):
        video_path = folder + "/" + all_video[i]
        print(video_path)
        img = cv2.imread(video_path)
        saveFaces(img, all_video[i], outputFolder)
