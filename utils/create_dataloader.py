import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import config as FIG
import cv2
import torch as nn

# 标准化
transform_BZ = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # 取决于数据集
    std=[0.229, 0.224, 0.225]
)
# transform_BZ = transforms.Normalize(
#     mean=[0.5,0.5,0.5],  # 取决于数据集
#     std=[0.5,0.5,0.5]
# )

# before_picture_size = 224
# after_picture_size = 224
class LoadDate(Dataset):
    def __init__(self,txt_path,picture_size,train_flag = True):
        self.img_info = self.get_imgs(txt_path)
        self.train_flag = train_flag
        self.picture_size = picture_size

        self.train_tf = transforms.Compose([
            transforms.Resize([self.picture_size,self.picture_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transform_BZ
        ])
        self.val_tf = transforms.Compose([
            transforms.Resize([self.picture_size,self.picture_size]),
            transforms.ToTensor(),
            transform_BZ
        ])


    def get_imgs(self,txt_path):
        with open(txt_path,'r',encoding='UTF-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'),imgs_info))
        return imgs_info

    def padding_black(self,img):
        w,h = img.size
        scale = self.picture_size/max(w,h)
        img_fg = img.resize([int(x) for x in [w*scale,h*scale]])
        size_fg = img_fg.size
        size_bg = self.picture_size
        img_bg = Image.new("RGB",(size_bg,size_bg))
        img_bg.paste(img_fg,((size_bg-size_fg[0])//2,
                             (size_bg-size_fg[1])//2))
        img = img_bg
        return img

    def __getitem__(self, index):
        img_path,label = self.img_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        # img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
        return img, label

    def __len__(self):
        return (len(self.img_info))

if __name__ == '__main__':
    picture_size = FIG.picture_size
    train_dataset = LoadDate("../train.txt",picture_size=picture_size)
    print("数据个数",len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32,
                                               shuffle=True)
    for imge,label in train_loader:
        # cv2.imshow(imge)
        print(imge.shape)  # (batch size, channel, height, width)
        print(label)
        print(imge)

