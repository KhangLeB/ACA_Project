import torch
import torchvision

m = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
m.eval()

block = m.features[3]
print(block)
print("use_res_connect:", getattr(block, "use_res_connect", None))
