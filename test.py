import torch
import config as FIG
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from model.mobile_facenet import MobileFaceNet



class Test():
    def __init__(self,model_path,img_path,picture_size,out_features,classes,model_type,model_path_1):
        self.model_path = model_path
        self.img_path = img_path
        self.picture_size = picture_size
        self.out_features = out_features
        self.classes = classes
        self.model_type = model_type
        self.model_path_1 = model_path_1

    def test(self):
        # 设置预测设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using {}".format(device))

        # 加载模型
        if (self.model_type == 'resnet_101'):
            model = torchvision.models.resnet101(pretrained=False)  # 默认为Flase,设置True会下载预训练文件
            net_dict = model.state_dict()
            model.load_state_dict(net_dict, strict=False)
            # torchvision 的resnet输出分类为1000中，根据自己的任务修改分类数量
            model.fc = torch.nn.Linear(2048, self.out_features)
            model = model.to(device)

        if(self.model_type == 'resnet_50'):
            model = torchvision.models.resnet50(pretrained=False)  # 默认为Flase,设置True会下载预训练文件
            net_dict = model.state_dict()
            model.load_state_dict(net_dict, strict=False)
            # torchvision 的resnet输出分类为1000中，根据自己的任务修改分类数量
            model.fc = torch.nn.Linear(2048, 128)
            model.fc_1 = torch.nn.Linear(128,self.out_features)
            model = model.to(device)
            # print(model)


        if(self.model_type == 'resnet_34'):
            model = torchvision.models.resnet34(pretrained=False)  # 默认为Flase,设置True会下载预训练文件
            net_dict = model.state_dict()
            model.load_state_dict(net_dict, strict=False)
            # torchvision 的resnet输出分类为1000中，根据自己的任务修改分类数量
            model.fc = torch.nn.Linear(512, self.out_features)
            model = model.to(device)

        if (self.model_type == 'resnet_18'):
            model = torchvision.models.resnet18(pretrained=False)  # 默认为Flase,设置True会下载预训练文件
            net_dict = model.state_dict()
            model.load_state_dict(net_dict, strict=False)
            # torchvision 的resnet输出分类为1000中，根据自己的任务修改分类数量
            model.fc = torch.nn.Linear(512, self.out_features)
            model = model.to(device)

        if(self.model_type == 'mobilefacenet'):
            model = MobileFaceNet()
            # model.linear = torch.nn.Linear(512,self.out_features)
            # model.bn = torch.nn.BatchNorm1d(self.out_features)
            model = model.to(device)
            print(model)


        if device == 'cpu':
            model.load_state_dict(torch.load(self.model_path_1,map_location=torch.device('cpu')))
            # model = torch.load(self.model_path,map_location=torch.device('cpu'))
            # model.to(device)
        else:
            model.load_state_dict(torch.load(self.model_path))

        # 加载图片
        image_path = self.img_path
        picture_size = self.picture_size  # 图片尺寸大小

        #  图片标准化
        transform_BZ = transforms.Normalize(
            # mean=[0.485, 0.456, 0.406],  # 取决于数据集
            # std=[0.229, 0.224, 0.225]
            mean=[0.5, 0.5, 0.5],  # 取决于数据集
            std=[0.5, 0.5, 0.5]
        )

        val_tf = transforms.Compose([transforms.Resize([picture_size,picture_size]),
                                     transforms.ToTensor(),
                                     transform_BZ
                                     ])


        def padding_black(img):
            w, h = img.size
            scale = picture_size / max(w, h)
            img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
            size_fg = img_fg.size
            size_bg = picture_size
            img_bg = Image.new("RGB", (size_bg, size_bg))
            img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                                  (size_bg - size_fg[1]) // 2))
            img = img_bg
            return img

        def save_feature(pre_feature_ndarray):
            # 保存全连接层的特征向量
            # pre_feature_ndarray = model.linear.state_dict()["weight"]
            # print(len(pre_feature_ndarray[0]))
            # print(pre_feature_ndarray)
            pre_feature_ndarray.data.cpu().numpy()
            pre_feature_ndarray = pre_feature_ndarray.tolist()
            feature = [str(i) for i in pre_feature_ndarray]
            str1 = ",".join(feature)
            with open("feature_txt/test_feature.txt", 'w') as f:
                f.write(str1)
                f.close()

        img = Image.open(image_path)
        img = img.convert('RGB')
        img = padding_black(img)
        # print(type(img))  # 打印输出图片的类型

        img_tensor = val_tf(img)
        # print(type(img_tensor))  #打印输出标准化后的图片类型

        # 增加图片的维度，之前是三维，模型要求四维
        img_tensor = Variable(torch.unsqueeze(img_tensor,dim=0).float(),requires_grad=False).to(device)

        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (picture_size, picture_size))
        # img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)/255
        # print(len(img_tensor))
        # 进行数据输入和模型转换
        model.eval()
        print(model)
        with torch.no_grad():
            feature = model(img_tensor)
            # print(output_tensor)
            print(feature)
            print(img_tensor)

            # 保存特征向量
            save_feature(feature)
            # # 将输出通过softmax变为概率值
            # output = torch.softmax(output_tensor,dim=1)
            # # print(output)
            #
            # # 输出可能性最大的那位
            # pred_value,pred_index = torch.max(output,1)
            #
            # print(pred_value)
            # print(pred_index)
            #
            # # 将数据从cuda转回cpu
            # if  torch.cuda.is_available() == False:
            #     pred_value = pred_value.detach().cpu().numpy()
            #     pred_index = pred_index.detach().cpu().numpy()
            #
            # # 类别标签
            # classes = self.classes
            # # 输出预测
            # print("预测类别：",classes[pred_index[0]],"概率为:",pred_value[0]*100,"%")


if __name__ == '__main__':
    classes = FIG.classes
    picture_size = FIG.picture_size
    out_features = FIG.out_features
    model_type = FIG.model_type
    model_path = FIG.model_path
    classes = FIG.classes
    model_path_1 = FIG.test_feature_model_path
    img_path = r'D:\python\pycharm project\deep_learning\face\face_attendance\data\zhazha\1.jpg'
    test = Test(model_path=model_path,picture_size=picture_size,img_path=img_path,out_features=out_features
                ,classes=classes,model_type=model_type,model_path_1=model_path_1)
    test.test()

