import os
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image


# 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 导入权重文件路径
weight_path = 'weights/unet.pth'
data_path = 'data/track_image'
save_path = 'data/train_image'

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=2, shuffle=True)
    net = UNet().to(device)

    # 导入权重文件
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    # 优化器和损失函数
    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    # 开始训练
    epoch = 1
    while True:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            # 计算损失值
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            # 优化
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # 显示训练次数
            if i % 5 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
            if i % 50 == 0:
                torch.save(net.state_dict(), weight_path)

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')

        epoch += 1
