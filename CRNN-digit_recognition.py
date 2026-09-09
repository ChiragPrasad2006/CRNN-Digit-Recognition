from torchvision.datasets import ImageFolder
from torch import transpose
from torch.nn import ReLU
from torch.nn import MaxPool2d
import torch
import torchvision
from torchmetrics import Accuracy
import pandas as pd
from torch import nn
from torchvision import transforms
import tqdm

torch.manual_seed(0)
class DigitRecognitionCRNN(nn.Module):
    def __init__(self,input_shape:int, hidden_shape:int, output_shape:int):
        super().__init__()
        self.block_1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_shape,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape,out_channels=hidden_shape,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_shape,out_channels=hidden_shape,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape,out_channels=hidden_shape,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.spatial_flatten=nn.Flatten(start_dim=2,end_dim=3)  # important as CNN reverses the order of [batch,channels,height*width] while RNN takes in [batch, height*width,channels]
        self.block_3=nn.Sequential(
            nn.RNN(input_size=hidden_shape,hidden_size=hidden_shape,num_layers=3,batch_first=True,nonlinearity='relu'), #3 RNN layers using ReLU
        )
        self.block_4=nn.Linear(in_features=hidden_shape,out_features=output_shape)

    def forward(self,x):
        x=self.block_1(x)
        x=self.block_2(x)
        x=self.spatial_flatten(x)
        x=x.transpose(1,2)
        out,_=self.block_3(x)
        out=out[:, -1, :]
        x=self.block_4(out)
        return x


#Transform dataset/define Transformers: size, num_output_channels and convert to tensor
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),  
                                transforms.Resize((64,64)),
                                transforms.ToTensor()])

#Load dataset
full_dataset = ImageFolder(root="data",transform=transform)
print(len(full_dataset))
print("class mapping:", full_dataset.class_to_idx)

#split dataset
train_size = int(0.8*len(full_dataset))
test_size = len(full_dataset) - train_size

train_data, test_data = torch.utils.data.random_split(full_dataset,[train_size,test_size],generator = torch.Generator().manual_seed(0))
print(len(train_data))
print(len(test_data))

#wrap into dataloaders:
train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=32,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=32,shuffle=False)

images, labels = next(iter(train_loader))
print(f"Batch image shape: {images.shape}")  # [32, 1, 28, 28]
print(f"Batch labels shape: {labels.shape}")  # [32]

torch.manual_seed(0)
model_CRNNv1 = DigitRecognitionCRNN(input_shape=1,hidden_shape=128,output_shape=10)

rand_image_tensor = torch.rand(size=(1,64,64))
model_CRNNv1(rand_image_tensor.unsqueeze(0))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_CRNNv1.parameters(),lr=0.1)
accuracy_fn = Accuracy(task="multiclass",num_classes=len(full_dataset.class_to_idx))

def print_train_test_time(start: float, end: float):
    total_time = end - start 
    print(f"train time on cpu: {total_time:.3f} seconds")
    return total_time

# train/test model

epochs =1
