# video testing
# read a video path then tell whether this video is raw or fake.
# Before running, a model must be trained in advance.
#

import os
from os.path import join
import subprocess
import cv2
from PIL import Image
import numpy as np
import argparse
from tensorflow import keras

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-m', '--model', type=str,default='Xception')
    parser.add_argument('-s', '--savePictures', type=bool, default=False)
    return parser.parse_args()


def main(args):
    modelsList = {'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'}
    video_name = args.path.split('/')[-1].split('.')[0]
    outputPath = "temp/"+video_name
    if not os.path.isdir('temp'):
        os.mkdir('temp')
    if os.path.isdir(outputPath):
        for i in os.listdir(outputPath):
            os.remove(outputPath+'/'+i)
        os.removedirs(outputPath)

    os.mkdir(outputPath)
    extract_frames(args.path.split('/')[-1], 0, outputPath, 'cv2')
    pictures = os.listdir(outputPath)
    data = []
    for picture in pictures:
        try:
            img = Image.open(outputPath+ '/' + picture)
            img_ndarray = np.asarray(img, dtype='float64') / 255
            data.append(img_ndarray)
        except ValueError as e:
            pass
    data = np.array(data).reshape(-1,299,299,3)
    for modelName in modelsList:
        print(modelName)
        fakeCount = 0
        rawCount = 0
        model = keras.models.load_model("models/" + modelName+ '/epoch' + str(10) + '.h5')
        result = model.predict(data)
        for i in result:
            prediction = np.argmax(i)
            if prediction == 1:
                fakeCount += 1
            if prediction == 0:
                rawCount += 1
        if fakeCount > rawCount:
            finalDecision = 1
        else:
            finalDecision = 0
        print("fakeCount=",fakeCount,"rawCount=", rawCount, "finalDecision=",)

def extract_frames(data_path, video_num = 0, folder_name = "raw", method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    # os.makedirs(output_path, exist_ok=True)
    if method == 'ffmpeg':
        output_path = "picture/raw"
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        count = 0
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            count += 1
            if count % 4 != 0:
                continue
            frame_num = saveFaces(image, video_num, frame_num, folder_name)
            # frame_num += 1
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))

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

def saveFaces(img, video_num, frame_num, full_path):
    """Method to crop the detected face from the picture"""
    img_size = 299
    faces = detectFaces(img)
    os.makedirs(full_path, exist_ok=True)

    if faces:
        frame_num += 1

    for (x1, y1, x2, y2) in faces:
        file_name = '{:05d}.png'.format(frame_num)
        video_name = '{:05d}-'.format(video_num)
        file_name = video_name + file_name

        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        img.crop((x1, y1, x2, y2)).resize((img_size, img_size), Image.LANCZOS).save(
            full_path + "/" + file_name)
    return frame_num

def is_video(name):
    """Method to judge whether the type of the video is right or not"""
    return (name[-4:] in ['.mp4'])


if __name__ == '__main__':
    args = parse_args()
    print(args.path)
    main(args)
