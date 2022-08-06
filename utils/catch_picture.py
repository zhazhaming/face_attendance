import cv2
import dlib
import os
import random
import config as FIG


# 图片存储位置
user_name = FIG.catch_picture_username
output_dir = FIG.catch_picture_path+user_name+'/'
size = FIG.catch_picture_size


# 设置截取图片的数量
index = 1
picture_num = FIG.catch_picture_num

# 设置是否改变截图图片亮度
islight = FIG.catch_picture_relight

# 判断是否有改文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 改变图片的亮度和对比度
def relight(img,light=1,bias=0):
    w = img.shape[0]
    h = img.shape[1]

    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):      # RGB三通道
                tmp = int(img[i,j,c]*light+bias)
                if tmp>255:
                    tmp=255
                elif tmp <0:
                    tmp = 0
                img[i,j,c] = tmp
    return img

detector = dlib.get_frontal_face_detector()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    if(index<=picture_num):
        print('正在截取低%s张图片' %index)
        # 摄像头读取图片
        success , img = cap.read()
        # 灰度化
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 调用检测器
        dets = detector(gray_img,1)
        for i ,d in  enumerate(dets):
            print(type(d))
            x1 = d.top() if d.top()>0 else 0
            y1 = d.bottom() if d.bottom() else 0
            x2 = d.left() if d.left()>0 else 0
            y2 = d.right() if d.right() >0 else 0

            # 截取脸部图片
            face = img[x1:y1,x2:y2]
            # 调整图片的对比度和亮度
            if (islight):
                face = relight(face,random.uniform(0.5,1.5),random.randint(-50,50))
            # 调整图片的大小
            face = cv2.resize(face,(size,size))
            print(type(face))
            # 显示图片
            cv2.imshow('image',face)
            # 保存图片
            cv2.imwrite(output_dir+'/'+str(index)+'.jpg',face)
            # index 自增
            index+=1
        key = cv2.waitKey(60)&0xff
    if index>picture_num:
        print('已经采集完成')
        cap.release()
        cv2.destroyAllWindows()
        break