import os
import json
import random
import numpy as np
import torch
from torchmetrics.classification import BinaryAccuracy
from torch import nn,optim
import torch.nn.functional as F
from torchvision import models
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Класс для остановки сети если не будет уменьшаться ошибка
class EarlyStopping:
  def __init__(self,path,name_model,patience=4,min_delta=0):
    self.name_model = name_model
    self.path = path
    self.patience = patience
    self.min_delta = min_delta
    self.file_saved = os.path.join(self.path,self.name_model,'params_model.pt')
    self.counter = 0
    self.early_stop = False
    self.best_loss = None
  def __call__(self,loss,model):
    if  self.best_loss is None:
      self.best_loss = loss
      self.save_checkpoint(model)
      print('First save')
    elif loss > self.best_loss - self.min_delta:
      self.counter += 1
      print(f'EarlyStopping counter {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_loss = loss
      self.save_checkpoint(model)
      self.counter = 0
      print('Loss was decreased')
  def save_checkpoint(self,model):
    file_parma = os.path.join(self.path,self.name_model)
    os.makedirs(file_parma,exist_ok=True)
    trianable_weights = {name:param for name,param in model.named_parameters()
    if param.requires_grad}
    torch.save(trianable_weights,self.file_saved)
    print(f'Parmas of {self.name_model} successfully saved')

class ContinueTrain:
  def __init__(self,file_path):
    self.file_path = file_path
  def load_states(self,model,optim):
    chekpoint = torch.load(self.file_path, map_location=next(model.parameters()).device)
    model.load_state_dict(chekpoint['model_state_dict'])
    optim.load_state_dict(chekpoint['optimizer_state_dict'])
    start_epoch = chekpoint['epoch']
    print('Continue training')
    return model, optim, start_epoch

#Тренировка моделей
class Train_Model:
  def __init__(self,name_model,path,epochs,train_data,val_data):
    self.name_model = name_model
    self.path = path
    self._epochs = epochs
    self.train_data = train_data
    self.val_data = val_data
    self.running_loss = 0.0
    self.__start_epoch = 0
    self.history_my_modelaccu = {'train_acc':[],
                                  'valid_acc':[]}
    self.history_chekpoint_path = os.path.join(self.path,self.name_model)
    self.early_stopper_val = EarlyStopping(path=self.path,name_model=self.name_model)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.criterion = nn.BCEWithLogitsLoss()
  @property
  def epochs(self):
    return self._epochs

  @epochs.setter
  def epochs(self,val):
    if val <= 0:
      raise ValueError('The number of epochs must be greater than zero')
    self._epochs = val
    print(f'Changed to {val}')

  def __call__(self,model):
    print(f'Training model {self.name_model}')
    model = model.to(self.device)
    low_name = self.name_model.lower()
    if 'resnet' in low_name:
      optimizer = optim.SGD([
        {'params':filter(lambda p: p.requires_grad, model.layer4.parameters()), 'lr':1e-4,
         'momentum':0.99, 'nesterov':True, 'weight_decay':0.23},
        {'params':filter(lambda p: p.requires_grad, model.fc.parameters()), 'lr':0.03,
         'momentum':0.98, 'nesterov':True, 'weight_decay':0.0}])
    elif 'my' in low_name:
      optimizer = optim.SGD(params=model.parameters(), lr=0.03,
                            momentum=0.98, nesterov=True, weight_decay=0.22)
    model.train()
    train_acc_metric = BinaryAccuracy().to(self.device)
    #Загрузка данных мадели для продолжения тренировки
    #cont_train = ContinueTrain(self.folder_chekpoint)
    #model, optim, self.__start_epoch = cont_train.load_states(model=model,optim=optimizer)

    for e in range(self.__start_epoch+1,self._epochs+1):
      print('\nSTART NEW EPOCH')
      running_loss = 0.0
      train_acc_metric.reset()
      pbar_train = tqdm(self.train_data,desc=f'Epochs = {e}/{self._epochs}',
                        ncols=200, colour='green')
      for imagen,labels in pbar_train:
        imagen,labels = imagen.to(self.device),labels.to(self.device)
        optimizer.zero_grad()
        outputs = model(imagen)
        loss = self.criterion(outputs.reshape(-1,1),labels.reshape(-1,1).float())
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        #Накапливает данные
        train_acc_metric.update(preds.reshape(-1,1),labels.reshape(-1,1).float())

        current_acc = train_acc_metric.compute() # Вычесляет точность модели
        running_loss += loss.item()
      epoch_accu = train_acc_metric.compute()
      avg_loss = running_loss/len(self.train_data)
      print(f'Epoch AVG loss = {avg_loss:.4f}, Epoch_Accuracy = {epoch_accu:.3f}')
      avg_val_loss = self.valid(model=model)
      self.early_stopper_val(loss=avg_val_loss,model=model)
      #Сохранение состаяние на данной эпохе в случае прерывания
      self.chekpoint_save(model=model,epoch=e,optim=optimizer,
                     loss=running_loss)
      self.history_my_modelaccu['train_acc'].append(epoch_accu.item())
      if self.early_stopper_val.early_stop:
        print('Early stopper triggered')
        break
    self.save_accuracy(history=self.history_my_modelaccu,
                       path=self.history_chekpoint_path)

    # Валидация
  def valid(self,model):
    model.eval()
    with torch.no_grad():
      correct = 0
      total = 0
      run_loss = 0.0
      progress_bar_valid = tqdm(self.val_data,desc=f'Valid ResNet50',
                                colour='red')
      for imagen,labels in progress_bar_valid:
        imagen, labels = imagen.to(self.device), labels.to(self.device)
        outputs = model(imagen).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        loss = self.criterion(outputs.reshape(-1,1),labels.reshape(-1,1).float())
        run_loss += loss.item()
        total += labels.size(0)
        correct += (preds == labels).sum().item()
      avg_val_loss = run_loss/len(self.val_data)
      accuracy_valid = correct/total
      accuracy_valid_persent = 100*accuracy_valid
      self.history_my_modelaccu['valid_acc'].append(accuracy_valid)
    print(f'{self.name_model} Valid Accuracy = {accuracy_valid_persent:.3f}%')
    return avg_val_loss
   # Сохраняет новое значение точности модели
  def save_accuracy(self,history,path):
    os.makedirs(path,exist_ok=True)
    save_file_js = os.path.join(path,'accuhistory.json')
    with open(save_file_js,'w') as f:
      json.dump(history,f)
    print('History Accuracy Saved Ok')
   #Сохранение состаяние на данной эпохе в случае прерывания
  def chekpoint_save(self,model,epoch,optim,loss):
    os.makedirs(self.history_chekpoint_path,exist_ok=True)
    save_file = os.path.join(self.history_chekpoint_path,'chekpoint.pth')
    if (epoch == self._epochs
        or self.early_stopper_val.early_stop == True):
      os.remove(save_file)
      print(f'Folder chekpoint {save_file} deleted')
    else:
      chekpoint = {'epoch':epoch,
                  'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optim.state_dict(),
                  'loss':loss}
      torch.save(chekpoint,save_file)
      print(f'State {self.name_model} epoch {epoch} successfully saved')

#График точности модели для тренировки и валидации
def plot_accuracy(history_path,name_model):
  with open(history_path,'r') as f:
    dict_acc = json.load(f)

  plt.figure(figsize=(10,6))

  plt.plot(dict_acc['train_acc'],label='Train Line',color='b',marker='o')
  plt.plot(dict_acc['valid_acc'],label='Valid Line',color='darkolivegreen',marker='o')

  plt.title(f'Accuracy epochs {name_model}',fontsize=20)
  plt.xlabel('Epochs',fontsize=10)
  plt.ylabel('Accuracy',fontsize=10)

  plt.grid(True,linestyle='--',alpha=0.5) #Сетка на графике
  plt.legend()

  plt.show()



