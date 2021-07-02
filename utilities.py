from typing import Optional, Tuple, Callable

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np


def make_layer(block: Callable, n_layers: int) -> nn.Sequential:
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def init_weights(network: nn.Module) -> None:
    def init_func(module: nn.Module) -> None:
        if hasattr(module, 'weight'):
            torch.nn.init.normal_(module.weight.data, 0.0, 0.01)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0.0)

    network.apply(init_func)



def make_submission(model: nn.Module, infer_dataloader: DataLoader, 
					infer_lr_dir: str) -> None:

	for name, hr_image in tqdm(infer_dataloader):
	    hr_image = hr_image.to(device)
	    with torch.no_grad():
	        predictions = model(hr_image)
	    pred_image = (predictions.cpu()[0].clamp(0, 1).permute(1, 2, 0) * 255).numpy().astype("uint8")
	    pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
	    cv2.imwrite(os.path.join(infer_lr_dir, name[0]), pred_image)