# face_attendance
## 1、项目使用Mobilefacenet进行训练人脸表征模块和Dlib人脸检测（后续将开发MTCNN进行人脸检测）
## 2、项目使用人脸直接的余弦距离判断人脸的相似度
## 3、项目包含了训练过程，也可以在注册人脸后直接使用使用。
## 4、训练过程使用的训练数据集为CASIA-Webface数据集
## 5、使用步骤：
   ### <br>在config.py文件中进行设置拍摄人脸数量、路径、用户名</br>
   ### <br>运行utils文件下的create_picture.py进行人脸脸部截取</br>
   ### <br>运行create_feature.py进行人脸特征提取</br>
   ### <br>最后执行根目录下的video_demo.py进行人脸识别</br>

