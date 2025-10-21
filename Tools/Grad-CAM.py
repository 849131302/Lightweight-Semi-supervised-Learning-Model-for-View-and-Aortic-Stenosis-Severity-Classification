import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.nn.functional import softmax, interpolate
from torchvision import transforms
from torchvision.io.image import read_image
from torchvision.models import resnet18
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import SmoothGradCAMpp, LayerCAM,GradCAM
from torchcam.utils import overlay_mask

from models.lwn import SLW_Net_4VIEW, reparameterize_model
from models.repvit import repvit_m0_9

# model = resnet18(pretrained=True).eval()
# cam_extractor = SmoothGradCAMpp(model)

#model = SLW_Net_4VIEW()
model=repvit_m0_9(num_classes =4)
model = reparameterize_model(model)
resume = ('D:\CV\LWM-SSL\checkpoints\TMED-2-VIEW\REPVIT\\best_epoch_30_balance_acc_96.8890_model.pth')#D:\CV\LWM-SSL\checkpoints\TMED-2-VIEW\REPVIT\\best_epoch_37_balance_acc_97.4390_model.pth
state_dict = torch.load(resume)
model.load_state_dict(state_dict)
model=model.eval()
for param in model.parameters():
    param.requires_grad_(True)
#cam_extractor = SmoothGradCAMpp(model=model,target_layer=model.stages[-1].blocks[-1].conv_ffn[-1],input_shape=(3, 112, 112))
#cam_extractor = SmoothGradCAMpp(model=model,target_layer=model.stages[-1],input_shape=(3, 112, 112))
cam_extractor =GradCAM(model=model,target_layer=model.features[-1],input_shape=(3, 112, 112))


for name, param in model.named_parameters():
    print(name, param.requires_grad)
print(model)


#img_path ="D:\CV\AS\TMED\TMED-VIEW\\test\A2C\\715s1_27.png"
img_path ="D:\CV\AS\TMED\TMED-VIEW\\test\A4C\\1040s1_1.png"
#img_path ="D:\CV\AS\TMED\TMED-VIEW\\test\PLAX\\9s1_41.png"#4838s1_5.png
#img_path ="D:\CV\AS\TMED\TMED-VIEW\\test\PSAX\\456s1_3.png"

data_transform = transforms.Compose([
  transforms.Grayscale(num_output_channels=3),
  transforms.ToTensor(),
])
img = Image.open(img_path)
# Preprocess it for your chosen model
input_tensor = data_transform(img)
out = model(input_tensor.unsqueeze(0))#input_tensor.unsqueeze(0)的作用是在input_tensor的第0维（即最外层维度）上增加一个维度，将其转换成一个大小为1的批次。
print(out.squeeze(0).argmax().item())

cams = cam_extractor(out.squeeze(0).argmax().item(), out)#out.squeeze(0).argmax().item()获取模型预测的类别索引
# target_index = 0
# cams = cam_extractor(target_index, out)
for cam in cams:
  print(cam.shape)
for name, cam in zip(cam_extractor.target_names, cams):
    plt.imshow(cam.squeeze(0).numpy());
    plt.axis('off');
    plt.title(name);
    plt.show()
for name, cam in zip(cam_extractor.target_names, cams):
  result = overlay_mask(to_pil_image(input_tensor), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
  plt.imshow(result); plt.axis('off'); plt.title(name); plt.show()
cam_extractor.remove_hooks()