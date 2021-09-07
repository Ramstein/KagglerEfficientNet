import pandas as pd
import numpy as np


csv_file  = r'C:\Users\Ramstein\Downloads\train.csv'

# img_file= pd.read_csv(csv_file, index_col='diagnosis')
# # for i, j in img_file, classIndex:
# #     print(i, j)
#
# print(img_file)
import csv, cv2, os, numpy as np

with open(csv_file, 'r') as f:
  reader = csv.reader(f)
  csv_as_list = list(reader)

for i in range(len(csv_as_list) + 1):
    print(i)
#
# for i in range(len(csv_as_list)):
#     if img == csv_as_list[i][0]:
#         classIndex = csv_as_list[i][1]
#         break
#
# IMG_SIZE =1050
# path = r'C:\Users\Ramstein\PycharmProjects\Kaggler\comps_14774_536888_train_images_00cc2b75cddd.png'
# img_array = cv2.imread(path, cv2.IMREAD_COLOR)
# resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# reshaped_img_array = np.array(np.reshape(resized_img_array, (-1, IMG_SIZE, IMG_SIZE, 3)))


# print(resized_img_array)
#
#
# !pip install wget
#
# !kaggle competitions download -c aptos2019-blindness-detection
#
#
#
# path = '/content'
#
# files = ['train_images.zip', 'test_images.zip']
#
# '''DownloadingData required imports'''
# import os
# from zipfile import ZipFile
#
# for fileName in files:
#     with ZipFile(os.path.join(path, fileName), 'r') as zipObj:  # extracts in the sample_data/PetImages/Cat & Dog
#         # Extract all the contents of zip file in different directory
#         if os.path.exists(os.path.join(path, fileName.split('.')[0])):
#             os.mkdir(os.path.join(path, fileName.split('.')[0]))
#         print('Extracting: ', fileName)
#         zipObj.extractall(os.path.join(path, fileName.split('.')[0]))
#
# extractedTo = path
# print('----Extraction Done---- \npathToExtractedData: {}'.format(extractedTo))
#

