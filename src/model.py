import torch
import torch.nn as nn
from torch.nn import functional as F

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, images, targets=None):
        bs, c, h, w = images.size()
        print(bs, c, h, w)
        x = F.relu(self.conv_1(images))
        x = self.avg_pool1(x)
        print(x.size())
        x = F.relu(self.conv_2(x))
        x = self.avg_pool2(x)
        print(x.size())
        return x

if __name__ == "__main__":
    cap_model = CaptchaModel(21)
    img = torch.rand((1, 3, 75, 300))
    target = torch.randint(1, 10, (1,5))
    x = cap_model(img, target)
