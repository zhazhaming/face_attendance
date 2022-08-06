# ------------------- 训练测试参数设置 ----------------------
#  图片大小
picture_size = 112
#  训练模型
model_type = 'mobilefacenet'     # resnet_18/resnet_34/resnet_50/resnet_101/mobilefacenet
# 训练数据集
train_txt = 'train.txt'
# 测试数据集
test_txt = 'test.txt'
# batch_size
batch_size = 64
# 迭代训练次数
epoch = 10
# 学习率
lr = 0.01
# 分类数量
out_features = 10575
# 分类类别
classes = ['0'*10575]
# classes = ['a','b','c','d']
# 输出文件路径
result_path = 'result'
# 训练好的模型文件路径
model_path = 'weight/2022-07-11-18-27-41.pth'

# ---------------------- 其他设置 -----------------------
# 截取图片设置
catch_picture_path = "../data/"  # 路径
catch_picture_username = 'zhazha'  # 文件名
catch_picture_size = 112
catch_picture_num = 50  # 截取图片数量
catch_picture_relight = False   # 是否开启随机调整图片暗亮
# 用于提取特征向量的模型文件
create_feature_model_path_1 = '../weight/MobileFace_Net.pth'
test_feature_model_path = 'weight/MobileFace_Net.pth'
# 保存特征向量的文件路径
save_feature_path = '../feature_txt/feature.txt'

# ----------- 视频流测试设置 ----------
# 提取特征向量的文件路径
feature_path = 'feature_txt/feature.txt'
# 人脸阈值   越低则越严格
face_confidence = 0.32