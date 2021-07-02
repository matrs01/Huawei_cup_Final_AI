from typing import Optional, Tuple, Callable

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SRDataset(Dataset):
    def __init__(
            self,
            hr_dir: str,
            lr_dir: str,
            samples: list,
            crop_size: Optional[int] = None,
            length: Optional[int] = None) -> None:
        self._hr_dir = hr_dir
        self._lr_dir = lr_dir
        self._crop_size = crop_size
        self._length = length
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples) if not self._length else self._length

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self._samples[item % len(self._samples)]
        lr_image = cv2.imread(os.path.join(self._lr_dir, name))
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.imread(os.path.join(self._hr_dir, name))
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        if hr_image.shape != (lr_image.shape[0] * 2, lr_image.shape[1] * 2, lr_image.shape[2]):
            raise RuntimeError(f"Shapes of LR and HR images mismatch for sample {name}")

        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1).float() / 255.
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1).float() / 255.

        if self._crop_size is not None:
            x_start = random.randint(0, lr_image.shape[1] - self._crop_size)
            y_start = random.randint(0, lr_image.shape[2] - self._crop_size)

            lr_image = lr_image[
                       :,
                       x_start:x_start + self._crop_size,
                       y_start:y_start + self._crop_size]
            hr_image = hr_image[
                       :,
                       x_start * 2:x_start * 2 + self._crop_size * 2,
                       y_start * 2:y_start * 2 + self._crop_size * 2]
        return hr_image.to(device), lr_image.to(device)




class InferDataset(Dataset):
    def __init__(
            self,
            hr_dir: str,
            samples: list,
            length: Optional[int] = None) -> None:
        self._hr_dir = hr_dir
        self._length = length
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples) if not self._length else self._length

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self._samples[item % len(self._samples)]
        hr_image = cv2.imread(os.path.join(self._hr_dir, name))
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1).float() / 255.

        return name, hr_image.to(device)





class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32, bias: bool = True) -> None:
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf: int, gc: int = 32) -> None:
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
            self,
            input_nc: int = 3,
            output_nc: int = 3,
            nf: int = 64,
            nb: int = 8,
            gc: int = 32) -> None:
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(input_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, output_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = F.interpolate(fea, scale_factor=0.5, mode='nearest')

        fea = self.lrelu(self.upconv1(fea))

        fea = self.lrelu(self.upconv2(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        out += F.interpolate(x, scale_factor=0.5, mode="nearest")
        return out

