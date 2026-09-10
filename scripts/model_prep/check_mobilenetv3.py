import torch
import torchvision

m = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
m.eval()
n_params = sum(p.numel() for p in m.parameters())
print("params:", n_params)
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = m(x)
print("output shape:", tuple(y.shape))
