import np
from PIL import Image
import tensorflow
import keras_applications


def addFileWithLabel(path, startNumber, endNumber, label):
    data = np.empty((endNumber-startNumber, 299, 299, 3))
    data_label = np.empty((endNumber-startNumber))
    for i in range(startNumber, endNumber):

        #filePath = path + str(i) + ".png"
        filePath = path + "{:07d}.png".format(i)
        img = Image.open(filePath)
        img_ndarray = np.asarray(img, dtype='float64') / 255
        print(img_ndarray.shape)
        data[i-startNumber] = img_ndarray

        """data.append(np.array(img))"""
        data_label[i-startNumber] = label
    return (data, data_label)

(d1, l1) = addFileWithLabel("picture/fake/",1,80,1)

(d2, l2) = addFileWithLabel("picture/raw/", 1, 80, 0)
d = np.concatenate((d1,d2),axis = 0)
l = np.append(l1,l2)

(test, testLabel) = addFileWithLabel("picture/testset/",101,120,1)

model = tensorflow.keras.applications.Xception()
model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(d,l, epochs = 10)
model.evaluate(test,testLabel)

