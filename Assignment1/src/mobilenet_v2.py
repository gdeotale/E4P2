import torch
from torchvision.models import mobilenet_v2

model = mobilenet_v2(pretrained=True)
model.eval()

traced_model = torch.jit.trace(model, torch.randn(1,3,224,224))
traced_model.save('mobilenet_v2.pt')