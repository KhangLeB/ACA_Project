import torch
import torchvision

m = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
m.eval()

for i, block in enumerate(m.features):
    n_params = sum(p.numel() for p in block.parameters())
    print(f"--- features[{i}]: {block.__class__.__name__}  params={n_params}")
    for name, sub in block.named_modules():
        if isinstance(sub, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"    {name or '.'}: {sub.__class__.__name__} "
                  f"in={getattr(sub,'in_channels',getattr(sub,'in_features',None))} "
                  f"out={getattr(sub,'out_channels',getattr(sub,'out_features',None))} "
                  f"kernel={getattr(sub,'kernel_size',None)} groups={getattr(sub,'groups',None)}")
