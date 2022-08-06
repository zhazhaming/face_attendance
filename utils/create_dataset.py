import os
import random

train_ratio = 0.8
test_ratio = 1 - train_ratio
# 图片格式
picture_geshi_1 = 'png'
picture_geshi_2 = 'jpg'

rootdate = r"D:\python\pycharm project\deep_learning\face\face_attendance\data"
train_list, test_list = [],[]
data_list = []

class_flag = -1
for root, dirs, files in os.walk(rootdate):
    if (len(files)==0):
        class_flag = -1
    # print('root={}'.format(root))
    # print('dirs={}'.format(dirs))
    # print('files={}'.format(files))
    # print(root)
    for i in range(len(files)):
        if files[i].endswith(picture_geshi_1) | files[i].endswith(picture_geshi_2):
            data_list.append(os.path.join(root,files[i]))  # 把所有图片都加载到data_list
            print(len(data_list))

    for i in range(0,int(len(files)*train_ratio)):
        if files[i].endswith(picture_geshi_1) | files[i].endswith(picture_geshi_2):
            train_data = os.path.join(root,files[i])+'\t'+str(class_flag)+'\n'  # 把训练图片加载到train_data
            train_list.append(train_data)

    for i in range(0,int(len(files)*test_ratio)):
        if files[i].endswith(picture_geshi_1) | files[i].endswith(picture_geshi_2):
            test_data = os.path.join(root,files[i])+'\t'+str(class_flag)+'\n'  # 把测速图片加载到test_data
            test_list.append(test_data)

    class_flag +=1

# print(train_list)
print('训练集有{}个'.format(len(train_list)))
print('测试集有{}个'.format(len(test_list)))

random.shuffle(train_list)
random.shuffle(test_list)

with open("../train.txt", 'w', encoding='UTF-8') as f:
    for train_img in train_list:
        f.write(str(train_img))


with open("../test.txt", 'w', encoding='UTF-8') as f:
    for test_img in test_list:
        f.write(str(test_img))