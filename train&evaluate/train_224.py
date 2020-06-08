import argparse
import os
import time
import numpy as np
import tensorflow
import tensorflow as tf
from PIL import Image
from tensorflow import keras

"""This python script will create a generator to read the data."""
"""Then we will start training."""
"""We will train one epoch each time and save the model."""
"""Model used: MobileNet"""


trainDataSet = []
"""
def printMemory():
    info = psutil.virtual_memory()
    print(u'memory used', psutil.Process(os.getpid()).memory_info().rss)
    print(u'total memory', info.total)
    print(u'memory percentage', info.percent)
"""

"""parse arguments from command line"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="Deepfakes")
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-w', '--workers', type=int, default=1)
    arg = parser.parse_args()
    return arg


"""load the model of the certain epoch"""


def model_load(target, epochNum):
    if not os.path.isdir("models_224/" + target):
        os.mkdir('models_224/' + target)
    if epochNum == 1:
        model = tf.keras.applications.MobileNet()
    else:
        model = keras.models.load_model("models_224/" + target + '/epoch' + str(epochNum - 1) + '.h5')
    return model


"""save the model"""


def model_save(model, target, epochNum):
    if not os.path.isdir("models_224/" + target):
        os.mkdir('models_224/' + target)
    model.save("models_224/" + target + '/epoch' + str(epochNum) + '.h5')


def main(args):
    print("Start time =  ", time.time())

    """open the train data paths"""
    trainDataFile = open("dataSetDict_224/" + args.target + "/trainDataSet.txt")
    while True:
        line = trainDataFile.readline()
        if not line:
            break
        record = line.split(" ")
        trainDataSet.append((record[0], int(record[1])))
    print("Training Data Paths loaded: number = ", len(trainDataSet))

    """produce the generator and the data"""
    data = tensorflow.data.Dataset.from_generator(dataGen, (tf.float32, tf.int32),
                                                  (tf.TensorShape([224, 224, 3]), tf.TensorShape([])))
    data = data.batch(args.batch)

    startTime = time.time()
    print("Start time =  ", time.time())

    """start training"""
    model = model_load(args.target, args.epoch)
    model.compile(optimizer='adam',
                  loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 3. train models
    if args.workers > 1:
        model.fit(data, epochs=1, use_multiprocessing=True, workers=args.workers)
    else:
        model.fit(data, epochs=1)
    model_save(model, args.target, args.epoch)

    print("Time used = ", time.time() - startTime)


"""generator of training data"""


def dataGen():
    for data in trainDataSet:
        path, label = data
        img = Image.open(path)
        img_ndarray = np.asarray(img, dtype='float64') / 255
        img_ndarray.resize(224, 224, 3)
        yield ((img_ndarray, label))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
