import argparse
import os
import random
import numpy as np
from PIL import Image
import tensorflow
import time
from tensorflow import keras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default="Deepfakes")
    parser.add_argument('-m', '--model', type=str, default="model.h5")
    parser.add_argument('-l', '--log', type=str, default="TrainLog.json")
    arg = parser.parse_args()
    return arg


def find_file(name="model.h5"):
    all_file = os.listdir("model")
    if name in all_file:
        model = keras.models.load_model("model/" + name)
    else:
        model = tensorflow.keras.applications.Xception()

    return model


def read_log(name = "log.txt"):
    file = open("log/"+name, "r")
    lines = file.readlines()
    latest_log = lines[-1]
    last_num = latest_log.split(" ")[1].split("\\")[0]
    new_first = int(last_num) + 1
    new_last = new_first + 49
    file.close()

    return new_first, new_last


def write_log(name = "log.txt", first = 1, last = 50):
    file = open("log/" + name, "a")
    output = str(first) + " " + str(last) + "\n"
    file.write(output)
    file.close()


def splitSet(target, splitRate=1, logName = "log.txt"):
    videoNumber = {
        "raw": 50,
        "Deepfakes": 50,
    }
    if os.path.exists("log/" + logName):
        first, last = read_log(logName)
    else:
        first = 1
        last = videoNumber[target]

    trainLength = round(videoNumber[target] * splitRate)
    validateLength = videoNumber[target] - trainLength
    # uList = list(range(1, videoNumber[target] + 1))
    uList = list(range(first, last + 1))
    vSet = random.sample(uList, validateLength)
    tSet = []
    for i in uList:
        if i not in vSet:
            tSet.append(i)
    write_log(logName, first, last)

    return {"name": target, "vSet": vSet, "tSet": tSet, "Num": videoNumber[target]}


def addFileWithLabel(path, split, label):
    # trainData = np.empty((0, 299, 299, 3))
    trainData = []
    # valData = np.empty((0, 299, 299, 3))
    valData = []
    trainLabel = np.empty(0)
    valLabel = np.empty(0)
    pictureList = os.listdir(path)
    pictureList = sorted(pictureList)

    for pictureName in pictureList:

        try:
            videoID = int(pictureName.split('-')[0])
            if videoID > split["Num"]:
                break
            pPath = path + pictureName
            with Image.open(pPath) as img:

                img_ndarray = np.asarray(img, dtype='float64') / 255

                if videoID in split["vSet"]:
                    valData.append(img_ndarray)
                    valLabel = np.append(valLabel, label)
                elif videoID in split["tSet"]:
                    trainData.append(img_ndarray)
                    trainLabel = np.append(trainLabel, label)
        except ValueError:
            pass

    trainData = np.array(trainData).reshape((-1, 299, 299, 3))
    valData = np.array(valData).reshape((-1, 299, 299, 3))
    # print(trainData.shape)
    """
    for i in range(startNumber, endNumber):

        filePath = path + str(i) + ".png"
        filePath = path + "{:04d}.png".format(i)
        img = Image.open(filePath)
        img_ndarray = np.asarray(img, dtype='float64') / 255
        print(img_ndarray.shape)
        data[i-startNumber] = img_ndarray
        data_label[i-startNumber] = label"""
    return (trainData, valData, trainLabel, valLabel)


def main(args):
    print("Start time =  ", time.time())

    # 1. create training set and validate set
    fakeSplit = splitSet(args.type, 0.8)
    rawSplit = splitSet("raw", 0.8)
    (train1, val1, trainLabel1, valLabel1) = addFileWithLabel("picture/fake/" + args.type + "/", fakeSplit, 1)
    (train2, val2, trainLabel2, valLabel2) = addFileWithLabel("picture/raw/", rawSplit, 0)
    # print("Start time =  ", time.time())
    trainData = np.concatenate((train1, train2), axis=0)
    trainLabel = np.append(trainLabel1, trainLabel2)
    valData = np.concatenate((val1, val2), axis=0)
    valLabel = np.append(valLabel1, valLabel2)

    startTime = time.time()
    print("Start time =  ", time.time())

    # 2. load existing model or create a new model
    model_name = args.model
    model = find_file(model_name)
    # model = tensorflow.keras.applications.Xception()
    model.compile(optimizer='adam',
                  loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 3. train model
    model.fit(trainData, trainLabel, epochs=10)
    model.save("model.h5")
    """model.evaluate(valData, valLabel)"""

    print("Time used = ", time.time() - startTime)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
