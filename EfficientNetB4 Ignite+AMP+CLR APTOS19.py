
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's# This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in
#
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#
# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#
# y_train_dir =
#
#
#
# # Any results you write to the current directory are saved as output. several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


x_train_dir = '/kaggle/input/train_images'
y_train_dir = '/kaggle/input/train.csv'

x_test_dir = '/kaggle/input/test_images'
y_test_dir = '/kaggle/input/test.csv'

y_train = pd.read_csv(y_train_dir)
y_test = pd.read_csv(y_test_dir)

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

import cv2
import pickle
import random
import datetime
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow._api.v1.image import resize

from tensorflow.python.keras.callbacks import TensorBoard

CATEGORIES ={0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative'}  #'''categories that we have to deal with'''

IMG_SIZE = 1050  #every image of 1050 X 1050
training_data=[]
x_train, y_train = [], []

def create_training_data():  # creating training datasets
    with open(y_train_dir, 'r') as f:
        reader = csv.reader(f)
        csv_as_list = list(reader)

    for img in os.listdir(x_train_dir):
        try:
            img_array = cv2.imread(os.path.join(x_train_dir, img), cv2.IMREAD_COLOR)
            resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            reshaped_img_array = np.array(np.reshape(resized_img_array, (-1, IMG_SIZE, IMG_SIZE, 3)))

#             print(reshaped_img_array)

            for i in range(len(csv_as_list)+1):

                if img.split('.')[0] == csv_as_list[i][0]:
                    print(i)
                    classIndex = csv_as_list[i][1]
                    training_data.append([reshaped_img_array, classIndex])
                    break
        except Exception as e:
            pass

create_training_data()
print(len(training_data))

for features, label in training_data:
    x_train.append(features)
    y_train.append(label)


pickle_out = open("x_train.pickle", 'wb')
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out= open('y_train.pickle', 'wb')
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_in = open('x_train.pickle', 'rb')
x_train = pickle.load(pickle_in)
pickle_in = open('y_train.pickle', 'rb')
y_train = pickle.load(pickle_in)

