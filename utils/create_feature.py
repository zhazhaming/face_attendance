import os
import torch
import numpy as np
import config as FIG
from PIL import Image
import torchvision.transforms as transforms
from model.mobile_facenet import MobileFaceNet
from torch.autograd import Variable



class create_feature():
    def __init__(self,model_path,img_path,catch_picture_size,model_type,feature_path):
        self.model_path = model_path
        self.img_path = img_path
        self.picture_size = catch_picture_size
        self.model_type = model_type
        self.feature_path = feature_path

    # 给图片加上黑边框
    def padding_black(self,img):
        w, h = img.size
        scale = self.picture_size / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = picture_size
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    # 保存特征向量
    # def save_feature(self,featureList,img_name):
    #     for i in range(len(featureList)):
    #         user_name = img_name[i]
    #         print(user_name)
    #         # feature_result.append(str(a) for a in featureList[i][0])
    #         feature_result = (str(a) for a in featureList[i][0])
    #         feature_result = ",".join(feature_result)
    #         # print(feature_result)
    #         with open("../feature.txt", 'a') as f:
    #             f.writelines(user_name)
    #             f.writelines(":")
    #             # f.writelines("[")
    #             f.writelines(feature_result)
    #             f.writelines('\n')
    #             # f.write('\n')
    #         # feature_result.clear()
    #     f.close()

    def save_feature_2(self,featureList,img_name):
        for i in range(len(featureList)):
            name = img_name[i]
            print(name)
            feature = str(featureList[i][0])
            with open(self.feature_path, 'a') as f:
                f.writelines(name)
                f.writelines(":")
                f.writelines(feature)
                f.writelines('\n')
        f.close()
    # 查找路径下的所有图片
    def search_picutre(self,img_path):
        imgList = []
        img_name = []
        dirs = os.listdir(img_path)
        # print(type(dirs))
        for file in dirs:
            img_name.append(file)
            pic_dir = os.path.join(img_path, file)
            # print(pic_dir)
            a = []
            for i in os.listdir(pic_dir):
                img_dir = os.path.join(pic_dir, i)
                a.append(img_dir)
            imgList.append(a)
        return imgList,img_name

    # 主函数
    def feature(self):
        # 设置预测设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using {} to catch feature".format(device))

        # 加载模型
        if (self.model_type == 'mobilefacenet'):
            model = MobileFaceNet()
            model = model.to(device)

        if device == 'cpu':
            model.load_state_dict(torch.load(self.model_path,map_location=torch.device('cpu')))
            # model = torch.load(self.model_path,False)
        else:
            model.load_state_dict(torch.load(self.model_path))


        picture_size = self.picture_size  # 图片尺寸大小

        #  图片标准化
        transform_BZ = transforms.Normalize(
            # mean=[0.485, 0.456, 0.406],  # 取决于数据集
            # std=[0.229, 0.224, 0.225]
            mean=[0.5, 0.5, 0.5],  # 取决于数据集
            std=[0.5, 0.5, 0.5]
        )

        # 图片预处理
        val_tf = transforms.Compose([transforms.Resize([picture_size, picture_size]),
                                     transforms.ToTensor(),
                                     transform_BZ
                                     ])
        featurelist = []
        Oneperson_feature = []
        # 加载图片
        img_path,img_name = create_feature.search_picutre(self,self.img_path)
        for pic_path in img_path:
            for image_path in pic_path:
                img = Image.open(image_path)  # 打开图片
                img = img.convert('RGB')  # 转换图片格式
                image = create_feature.padding_black(self,img)  # 为图片加上黑边
                img_tensor = val_tf(image)   # 图片预处理
                img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False).to(device)   # 增加图片的维度，之前是三维，模型要求四维

                # 进行数据输入和保存特征向量
                model.eval()
                with torch.no_grad():
                    features= model(img_tensor)
                    features = features.detach().numpy()  # 将tensor转为ndarry
                    Oneperson_feature.append(features)
            featuremean = np.array(Oneperson_feature).mean(axis=0)
            featurelist.append(featuremean.tolist())
            Oneperson_feature.clear()  # 计算完一个人的特征向量后清除，避免对后面计算产生影响
        # print(featurelist[0][0])
        create_feature.save_feature_2(self,featurelist,img_name)
        print("完成了{}个人的特征向量提取".format(len(featurelist)))

if __name__ == '__main__':
    picture_size = FIG.catch_picture_size
    model_type = FIG.model_type
    model_path = FIG.create_feature_model_path_1
    save_feature_path = FIG.save_feature_path
    img_path = r'D:\python\pycharm project\deep_learning\face\face_attendance\data'
    create_features = create_feature(model_path=model_path,catch_picture_size=picture_size,img_path=img_path,model_type=model_type,
                                     feature_path=save_feature_path)
    create_features.feature()
