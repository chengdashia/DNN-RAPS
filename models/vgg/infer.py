import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# 确保.pth文件的路径是正确的
model_path = 'vgg.pth'
image_path = '../../images/dog.jpg'

# 加载模型
model = models.vgg11()  # 确保这里的结构与你的.pth文件中的模型结构相匹配
model.load_state_dict(torch.load(model_path))

# 设置为评估模式
model.eval()

# 图片预处理
# 这里的预处理方式应与训练时使用的方式相同
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 打开图片
image = Image.open(image_path)

# 预处理图片
image = transform(image).unsqueeze(0)  # 增加一个维度，因为模型期望的是批量的数据

# 推理
with torch.no_grad():
    outputs = model(image)

# 获取最可能的类别
_, predicted = outputs.max(1)

# 输出预测结果
# 这里假设 'predicted' 是一个索引，你可能需要根据你的数据集将其转换为实际的类别名称
print(f'Predicted class index: {predicted.item()}')