import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x.unsqueeze(2) ** 2, dim=1) + self.eps)
        x = x / norm
        return x


def input_norm(x):
    data_mean = torch.mean(x, dim=[2, 3]).unsqueeze(-1).unsqueeze(-1)
    data_std = torch.std(x, dim=[2, 3]).unsqueeze(-1).unsqueeze(-1) + 1e-10
    data_norm = (x - data_mean.detach()) / data_std.detach()
    return data_norm


class conv(nn.Module):
    def __init__(self, inch, outch, kernal_size=3, stride=1, padding=0):
        super(conv, self).__init__()
        if padding is None:
            padding = (kernal_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(inch, outch, kernal_size, stride, padding, bias=False),
            nn.BatchNorm2d(outch, affine=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class desNet(nn.Module):
    def __init__(self, pad=False):
        super(desNet, self).__init__()
        if pad:
            first_conv = conv(1, 32, stride=2, padding=28)
        else:
            first_conv = conv(1, 32, stride=2)
        self.net = nn.Sequential(
            first_conv,
            conv(32, 32),
            conv(32, 64),
            nn.MaxPool2d(2),
            conv(64, 64),
            conv(64, 128),
            conv(128, 128),
            conv(128, 256),
            conv(256, 128),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128, affine=False)
        )

    def forward(self, x):
        x_t = input_norm(x)
        x_t = self.net(x_t)
        x = L2Norm()(x_t)
        return x


class Des:
    def __init__(self, device="cpu"):
        self.model = desNet()
        loaded_model = torch.load("../CES/net_100.pth")
        # use the next code when running locally
        # loaded_model = torch.load("model/net_100.pth", map_location=torch.device('cpu'))
        self.model.load_state_dict(loaded_model)
        self.model.eval()
        self.device = device
        self.model.to(device)

    def compute(self, image, kp):
        l = []
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # img = torch.from_numpy(image_gray.astype(np.float32))
        img = torch.from_numpy(image.astype(np.float32))
        img = img.to(self.device)
        dim = (32, 32, 32, 32)
        img = F.pad(img, dim, "constant", value=0)
        for pt in kp:
            x = int(pt.pt[0] + 32)
            y = int(pt.pt[1] + 32)
            img_t = img[x - 32:x + 32, y - 32:y + 32] / 255
            img_t = img_t.view(1, 1, img_t.size(0), img_t.size(1))
            desc = self.model(img_t)
            desc_numpy = desc.cpu().detach().float().squeeze().numpy()
            # desc_numpy = np.clip(((desc_numpy + 0.5) * 128).astype(np.int32), 0, 255).astype(np.uint8)
            l.append(desc_numpy)
        return l


if __name__ == "__main__":
    input_patch = np.random.random([64, 64]).astype(np.float32)
    kp = [cv2.KeyPoint(32, 32, 64)]
    des = Des()
    l = des.compute(input_patch, kp)
    print('done')


