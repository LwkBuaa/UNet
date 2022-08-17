import os
import cv2
import torch
import time
from net import *
import numpy as np
from utils.utils import keep_image_size_open
from data import *
from torchvision.utils import save_image
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# 使用cuda
net = UNet().cuda()

# 打开已有的参数
weights = 'weights/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully import weights!')
else:
    print('no loading weights')

# 用于存放中点的列像素
pi = []
# 用于存放中点的行像素值
pcenter = []

i = 0
directory_name = "data/test_image_mp4/else/"
k = 0
dataframe_list = []
for filename in os.listdir(directory_name):
    frame = cv2.imread(directory_name + filename)
    # (480, 640, 3)

    # 通道转换
    frame1 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # 图片统一大小
    frame1 = Image.fromarray(np.uint8(frame1))
    size = (256, 256)
    temp = max(frame1.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(frame1, (0, 0))
    mask = mask.resize(size)
    img = mask

    # 图片进网络
    img_data = transform(img).cuda()
    img_data = torch.unsqueeze(img_data, dim=0)
    out = net(img_data)

    # 维度转换
    out = out.permute(0, 2, 3, 1)
    # out.cpu().detach().numpy()
    out = np.array(out.cpu().detach().numpy())  # 很费时间
    out1 = (out[0] * 255).astype(np.uint8)

    # 计算中点
    j = 0
    for i in range(0, 192, 2):
        b = np.where(out1[i, :, 0] >= 120)
        if b[0] != []:
            b[0].sort()
            center = b[0][int(len(b[0]) / 2)]
            pi.append(i)
            pcenter.append(center)
            # 绘制中点
            cv2.circle(frame, (int(2.5 * pcenter[j]), int(2.5 * pi[j])), 4, (0, 0, 255), -1)

            j += 1

    rf1 = pd.DataFrame({'行像素值_图片{}'.format(k): pi, '列像素值_图片{}'.format(k): pcenter})
    dataframe_list.append(rf1)
    k += 1

    # 清除中点
    pcenter.clear()
    pi.clear()

    # 显示图片流
    # frame = frame[:,:,::-1]
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("image", out[0][:, :, ::-1])  # Write the frame onto the output.
    cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.imshow("image1", frame)  # Write the frame onto the output.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


data_all = pd.concat(dataframe_list, axis=1)
data_all.to_excel('results/point_center/point_center1.xlsx', index=False, encoding='utf-8')
