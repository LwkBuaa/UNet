import os
import numpy as np
import torch
import cv2
from net import *
from utils.utils import keep_image_size_open
from data import *
from torchvision.utils import save_image
import time
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# 使用cuda
net = UNet().cuda()

# 导入视频
cap = cv2.VideoCapture("data/test_image_mp4/1.mp4")
assert cap.isOpened()
x_shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
y_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
four_cc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("result", four_cc, 20, (x_shape, y_shape))


# 打开已有的参数
weights = 'weights/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully import weights!')
else:
    print('no loading weights')

ret, frame = cap.read()

# 用于存放中点的列像素
pi = []
# 用于存放中点的行像素值
pcenter = []

l = 0
dataframe_list = []
k = 0
while ret:
    # 初始时刻
    start_time = time.perf_counter()
    # 调节视频速度
    if k % 3 == 0:
        # 通道转换
        frame1 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 统一size
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

        # 结束时刻
        end_time = time.perf_counter()
        fps = 1 / np.round(end_time - start_time, 3)  # Measure the FPS.
        print(f"Frames Per Second : {fps}")
        show_fps = "fps:" + "{:.3}".format(fps)

        # 维度变换
        out = out.permute(0, 2, 3, 1)
        # out.cpu().detach().numpy()
        out = np.array(out.cpu().detach().numpy())   # 很费时间
        out1 = (out[0] * 255).astype(np.uint8)

        # 计算中点
        j = 0
        for i in range(0, 142, 2):
            b = np.where(out1[i, :, 0] >= 120)
            if b[0] != []:
                b[0].sort()
                center = b[0][int(len(b[0])/2)]
                pi.append(i)
                pcenter.append(center)
                # 绘制中点
                cv2.circle(frame, (int(7.5*pcenter[j]), int(7.5*pi[j])), 10, (0, 0, 255), -1)
                # 绘制fps
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, show_fps, (100, 100), font, 4, (0, 255, 0), 2, cv2.LINE_AA)
                j += 1

        rf1 = pd.DataFrame({'行像素值_帧{}'.format(l): pi, '列像素值_帧{}'.format(l): pcenter})
        dataframe_list.append(rf1)
        l += 1

        pcenter.clear()
        pi.clear()

        # 显示视频流
        # frame = frame[:,:,::-1]
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow("image", out[0][:,:,::-1])  # Write the frame onto the output.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.imshow("image1", frame)  # Write the frame onto the output.

    k += 1
    ret, frame = cap.read()

data_all = pd.concat(dataframe_list, axis=1)
data_all.to_excel('results/point_center/point_center2.xlsx', index=False, encoding='utf-8')

cap.release()
cv2.destroyAllWindows()
