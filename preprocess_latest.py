# Data preprocess - Neal
import os
from os.path import join
import subprocess
import cv2
from PIL import Image
import sys

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
        every25 = 0;
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            every25 += 1
            if (every25 % 4) != 0:
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

def saveFaces(img, video_num, frame_num, folder_name):
    """Method to crop the detected face from the picture"""
    img_size = 299
    faces = detectFaces(img)
    full_path = "picture/" + folder_name
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

    # add the terminal input into the arguments list
    arguments = sys.argv
    type = arguments[1]
    folder = arguments[2]
    already_videoes = sorted(os.listdir("picture/fake/" + folder))
    # print(already_videoes)
    last_picture = already_videoes[-1]
    lastvideo = last_picture.split('-')[0]
    # print(lastvideo)

    # executue according to the type
    if (type == "raw"):
        # operate the raw video into faces
        all_video = os.listdir("download/original_sequences/" + folder + "/c23/videos")
        video = [i for i in all_video if is_video(i)]
        video_num = 1
        outputFolder = "raw/" + folder
        for i in range(len(video)):
            video_path = "download/original_sequences/" + folder + "/c23/videos/" + video[i]
            print(video_path)
            # frame_folder = "real_frame"
            extract_frames(video_path, video_num, outputFolder)
            video_num += 1

    if (type == "fake"):
        # operate the fake video into faces
        all_video = os.listdir("download/manipulated_sequences/" + folder + "/c23/videos")
        video = [i for i in all_video if is_video(i)]
        video_num = 1
        outputFolder = "fake/" + folder
        for i in range(len(video)):
            if (video_num < int(lastvideo)):
                video_num += 1
                continue
            video_path = "download/manipulated_sequences/" + folder + "/c23/videos/" + video[i]
            print(video_path)
            # frame_folder = "real_frame"
            extract_frames(video_path, video_num, outputFolder)
            video_num += 1
