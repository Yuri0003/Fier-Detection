import numpy as np
import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

model_resnet50 = models.resnet50
base_weights = models.ResNet50_Weights.DEFAULT

# Изменение Верхней части модели ResNet50
def fine_turning_model(BaseModel,base_weights):
  base = BaseModel(weights=base_weights)
  in_features = base.fc.in_features
  # Заморозка слоёв в начале модели
  for param in base.parameters():
    param.requires_grad = False
  for param in base.layer4.parameters():
    param.requires_grad = True
  base.fc = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(0.3),
      nn.Linear(in_features,512),
      nn.ReLU(),
      nn.Linear(512,120),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(120,1))
  
  return base


#Создание 1 собственной модели CNN
#Блок обходной связи
class ResBlock(nn.Module):
  def __init__(self,input,out):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=input,out_channels=out,kernel_size=3,stride=1,padding=1)
    self.batch1 = nn.BatchNorm2d(out)
    self.conv2 = nn.Conv2d(out,input,kernel_size=3,stride=1,padding=1)
    self.batch2 = nn.BatchNorm2d(input)
  def forward(self,x):
    out_conv1 = self.conv1(x)
    out_norm1 = self.batch1(out_conv1)
    out1 = F.relu(out_norm1)
    out_conv2 = self.conv2(out1)
    out_norm2 = self.batch2(out_conv2)
    out2 = F.relu(out_norm2)
    return out2 + x

#Блок общей сети
class MyCNNR(nn.Module):
  def __init__(self,n_blocks=5,num_classes=1):
    super(MyCNNR,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=38,kernel_size=3,stride=1,padding=1)
    self.bn1 = nn.BatchNorm2d(38)
    self.conv2 = nn.Conv2d(in_channels=38,out_channels=64,kernel_size=3,stride=3,padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    self.resblocks = nn.Sequential(
        *[ResBlock(input=64,out=128) for _ in range(n_blocks)])
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(64,100)
    self.dropout = nn.Dropout(0.5)
    self.fc2 = nn.Linear(100,64)
    self.bn3 = nn.BatchNorm1d(64)
    self.fc3 = nn.Linear(64,32)
    self.fc_quality = nn.Linear(32,num_classes)
    self.apply(self._init_weights)
  def _init_weights(self,layer):
    if isinstance(layer,nn.Conv2d):
      nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
      if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    elif isinstance(layer,nn.BatchNorm2d):
      nn.init.constant_(layer.weight,0.5)
      nn.init.zeros_(layer.bias)
  def forward(self,x):
    out = self.pool(self.bn1(self.conv1(x)))
    out = F.relu(out)
    out = self.pool(self.bn2(self.conv2(out)))
    out = F.relu(out)
    out = self.resblocks(out)
    out_pool = self.avgpool(out)
    out = self.flat(out_pool)
    out = self.dropout(F.relu(self.fc1(out)))
    out = F.relu(self.bn3(self.fc2(out)))
    out = F.relu(self.fc3(out))
    out = self.fc_quality(out)
    return out

#Создание 2 собственной модели CNN
class MyCNN(nn.Module):
  def __init__(self,num_classes=1):
    super().__init__()
    # Свёрточные слои
    self.features = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(32,64,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(64,64,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Conv2d(64,128,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(128,150,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(150,150,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(150),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Conv2d(150,128,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(128,128,kernel_size=3,stride=3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.AdaptiveAvgPool2d((1,1)))

    # Полносвязная сеть из линейных слоёв
    self.classifire = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),

        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Linear(32,num_classes))

  def forward(self,x):
    x = self.features(x)
    x = self.classifire(x)
    return x

def parametrs(model):
  if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_model = model.to(device)
    summary(my_model,input_size=(3,244,244))
    print(f'Параметры готовой модели {sum(p.numel() for p in my_model.parameters())}')


if __name__ == 'main':
  model_res50 = fine_turning_model(BaseModel=model_resnet50,base_weights=base_weights) 
  res50_params = parametrs(model=model_res50)
  my_model_cnn = MyCNN(num_classes=1)
  cnn_parmas = parametrs(model=my_model_cnn)
  my_resmodel = MyCNNR(n_blocks=4,num_classes=1)
  my_res_params = parametrs(model=my_resmodel)
