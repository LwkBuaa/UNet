import os
from torch.utils.data import Dataset
from utils.utils import *
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor()
])


# 数据集导入
class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'json_file_train'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'json_file_train', segment_name)
        image_path = os.path.join(self.path, 'jpg_file_train', segment_name)
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image), transform(segment_image)


if __name__ == '__main__':
    data = MyDataset(r"data/track_image/track_image0")
    print("Importing the dataset succeeded")
