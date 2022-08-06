import ast
import time
import dlib
import cv2
import math
import torch
import numpy as np
import config as FIG
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
from model.mobile_facenet import MobileFaceNet

class video():
    def __init__(self,model_path,feature_path,picture_size,face_confidence):
        self.model_path = model_path
        self.feature_path = feature_path
        self.picture_size = picture_size
        self.face_confidence = face_confidence

    # 加载特征向量
    def load_feature(self):
        name_list = []
        feature_list = []
        with open(self.feature_path, 'r') as f:
            while (True):
                a = f.readline()
                if not a :
                    break
                name,features= a.split(':')
                name_list.append([name])
                feature = features.split('\n')
                a = feature[0]
                b = ast.literal_eval(a)
                feature_list.append(b)
        return name_list,feature_list

    # 获得视频流图片的特征向量
    def getVideoFeature(self,model,image,device):
        #  图片标准化
        transform_BZ = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],  # 取决于数据集
            std=[0.5, 0.5, 0.5]
        )
        # 图片预处理
        val_tf = transforms.Compose([transforms.Resize([self.picture_size, self.picture_size]),
                                     transforms.ToTensor(),
                                     transform_BZ
                                     ])
        image = Image.fromarray(image)
        img_tensor = val_tf(image)  # 图片预处理
        img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False).to(
            device)  # 增加图片的维度，之前是三维，模型要求四维

        model.eval()
        with torch.no_grad():
            features = model(img_tensor)
            features = features.detach().cpu().numpy()  # 将tensor转为ndarry

        return features

    # 计算两者距离
    def distance(self,feture_1,feture_2):
        # 两者之间的欧式公式
        # feture_1 = np.array(feture_1)
        # feture_2 = np.array(feture_2)
        # dist = np.sqrt(np.sum(np.square(feture_1-feture_2)))

        # 使用两者之间的余弦距离
        dist = np.dot(feture_1, feture_2) / (np.linalg.norm(feture_1) * np.linalg.norm(feture_2))
        dist = np.arccos(dist)/math.pi
        return dist

    # 主函数
    def eval(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MobileFaceNet()
        model.to(device)
        if device == "cpu":
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(self.model_path))
        detector = dlib.get_frontal_face_detector()
        cap = cv2.VideoCapture(0)
        probability = []
        while (True):
            # 摄像头读取图片
            success, img = cap.read()
            # 灰度化
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 调用检测器
            dets = detector(gray_img, 1)
            start_time = time.time()
            if dets is not None:
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    # 截取脸部图片
                    face = img[x1:y1, x2:y2]
                    # 改变图片大小
                    face = cv2.resize(face,(FIG.picture_size,FIG.picture_size))
                    features_1 = video.getVideoFeature(self,model,face,device)
                    name_list,feature_list = video.load_feature(self)
                    # 每张图片都和读取到的特征向量进行一次对计算
                    for i in range(len(feature_list)):
                        features_2 = feature_list[i]
                        features_2 = np.array(features_2)
                        num = video.distance(self,features_1,features_2)
                        probability.append(num)
                    # print(probability)  # 打印和各特征向量比例，越低相似度越高
                    min_pro = min(probability)  # 获取最低的比例
                    index = probability.index(min_pro)
                    user_name = name_list[index]
                    probability.clear() # 一张图片对比完成及时清除，避免对下一张图片造成影响
                    if min_pro <= self.face_confidence:
                        cv2.putText(img, str(user_name), (y2,y1-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(img, str("Unknow"), (y2, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    img = cv2.rectangle(img,(y2,y1),(x2,x1),(0, 255, 0), 2)
            end_time = time.time()
            use_time = end_time-start_time
            if (use_time != 0):
                FPS = 1/(end_time-start_time)
                cv2.putText(img,"PFS:{:.2f}".format(FPS),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            else:
                cv2.putText(img,"PFS:----" , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow("img",img)
            if cv2.waitKeyEx(1) == ord('q'):  # 如果按下‘q’则推出
                break


if __name__ == '__main__':
    # 参数设置
    feature_path = FIG.feature_path
    picture_size = FIG.picture_size
    face_confidence = FIG.face_confidence
    model_path = FIG.test_feature_model_path
    V = video(model_path=model_path,feature_path=feature_path,picture_size=picture_size,face_confidence=face_confidence)
    V.eval()