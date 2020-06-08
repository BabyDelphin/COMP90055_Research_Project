import argparse

import numpy as np
from PIL import Image
from tensorflow import keras

"""This python script will evaluate the model of the last epoch."""
"""picture size used = 224*224*3"""


testDataSet = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="Deepfakes")
    arg = parser.parse_args()
    return arg


def dataGen():
    for data in testDataSet:
        path, label = data
        img = Image.open(path)
        img_ndarray = np.asarray(img, dtype='float64') / 255
        img_ndarray.resize(1, 224, 224, 3)
        yield ((img_ndarray, label))


def main(args):
    testDataFile = open("dataSetDict_224/" + args.target + "/testDataSet.txt")
    while True:
        line = testDataFile.readline()
        if not line:
            break
        record = line.split(" ")
        testDataSet.append((record[0], int(record[1])))
    print("Test Data Paths loaded: number = ", len(testDataSet))

    print("testing epoch=", 10, '\n')
    model = keras.models.load_model("models_224/" + args.target + '/epoch' + str(10) + '.h5')
    model.evaluate(dataGen(), verbose=1)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
