# Face Detection and Crop - Neal (2020-04-14 1:25)
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import os


def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x + width, y + height))
    return result


def saveFaces(image_name):
    img_size = 299
    faces = detectFaces(image_name)
    if faces:
        save_path = image_name.split('.')[0]
        save_path = save_path.split('/')[1]
        count = 0
    for (x1, y1, x2, y2) in faces:
        file_name = save_path + "-" + str(count) + ".png"
        Image.open(image_name).crop((x1, y1, x2, y2)).resize((img_size, img_size), Image.LANCZOS).save(
            "out/" + file_name)
        count += 1


def is_inp(name):
    return (name[-4:] in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'])


if __name__ == '__main__':
    all_inps = os.listdir("in")
    all_inp = [i for i in all_inps if is_inp(i)]
    for i in range(len(all_inp)):
        path_ = "in/" + all_inp[i]
        print(path_)
        saveFaces(path_)
