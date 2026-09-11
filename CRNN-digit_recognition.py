from torchvision.datasets import ImageFolder
from torch import transpose
import torch
from torchmetrics import Accuracy
from torch import nn
from torchvision import transforms
from tqdm.auto import tqdm
from timeit import default_timer as timer
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
        self.spatial_flatten=nn.Flatten(start_dim=2,end_dim=3)  # important as CNN gives the order in [batch,channels,height*width] while RNN takes in [batch, height*width,channels]
        self.linear_1=nn.Linear(in_features=hidden_shape,out_features=hidden_shape)
        self.block_3=nn.GRU(input_size=hidden_shape,hidden_size=hidden_shape,num_layers=2,batch_first=True) #3 RNN layers using ReLU
        self.linear_2=nn.Linear(in_features=hidden_shape,out_features=output_shape)

    def forward(self,x):
        x=self.block_1(x)
        x=self.block_2(x)
        x=self.spatial_flatten(x)
        x=x.transpose(1,2)   # important as CNN reverses the order of [batch,channels,height*width] while RNN takes in [batch, height*width,channels]
        x=self.linear_1(x)
        out,_=self.block_3(x)
        out=out[:, -1, :]    # passing the output of RNN to Linear layer, since RNN gives output in tuples, we need to pass only the last time step's output to the linear layer which gives the final result
        x=self.linear_2(out)
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
train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=128,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=128,shuffle=False)

images, labels = next(iter(train_loader))
print(f"Batch image shape: {images.shape}")  # [32, 1, 28, 28]
print(f"Batch labels shape: {labels.shape}")  # [32]

torch.manual_seed(0)
model_CRNNv1 = DigitRecognitionCRNN(input_shape=1,hidden_shape=128,output_shape=10).to(device)

rand_image_tensor = torch.rand(size=(1,64,64)).to(device)
model_CRNNv1(rand_image_tensor.unsqueeze(0))

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(params=model_CRNNv1.parameters(),lr=0.001)
accuracy_fn = Accuracy(task="multiclass",num_classes=len(full_dataset.class_to_idx)).to(device)

def print_train_test_time(start: float, end: float, device=None):
    total_time = end - start 
    print(f"train time on {device}: {total_time:.3f} seconds")
    return total_time

# train/test model

print_train_test_time_start = timer()
epochs =100
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n")
    train_loss=0
    for batch, (X,y) in enumerate(train_loader):
        model_CRNNv1.train()
        X,y=X.to(device),y.to(device)
        y_pred = model_CRNNv1(X)
        loss=loss_fn(y_pred,y)
        train_loss +=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch+1)%400==0:
            print(f"looked at {batch*len(X)}/{len(train_loader.dataset)}")
    train_loss /= len(train_loader)

    test_loss, test_acc = 0, 0
    model_CRNNv1.eval()
    with torch.inference_mode():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_pred = model_CRNNv1(X_test)
            test_loss += loss_fn(test_pred, y_test).item()
            test_acc += accuracy_fn(test_pred.argmax(dim=1), y_test).item()
        test_acc /= len(test_loader)
        test_loss /= len(test_loader)
    print(f"train loss: {train_loss:.5f}")
    print(f"test loss: {test_loss:.5f}")
    print(f"test accuracy: {test_acc:.4f}")

print_train_test_time_end = timer()
total_time = print_train_test_time(start=print_train_test_time_start, end=print_train_test_time_end, device=device)

MODEL_PATH=Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)
MODEL_NAME="CRNN_digit_recognition_v1.pth"
MODEL_SAVE_PATH =MODEL_PATH / MODEL_NAME

print(f"saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_CRNNv1.state_dict(),f=MODEL_SAVE_PATH)

