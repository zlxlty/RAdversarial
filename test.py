import torchvision
import torch
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import numpy as np
import torchvision.models as models
import time
import datetime

transform_size = 227

transform = transforms.Compose([
    transforms.Resize((transform_size, transform_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



def make_prep_folder(dataset_folder):
    tf = "transformed_folder"
    

    classes = []
    f = open(os.path.join(dataset_folder, "labels.txt"), "r")   
    lines = f.readlines()
    f.close()
    classes = {}
    for line in lines:
        line = line.split(": ")
        image_name = line[0]
        image_class = line[1]
        if image_class not in classes:
            classes[image_class] = []
        classes[image_class].append(image_name)

    try:
        shutil.rmtree(tf)
    except:
        pass
        
    os.mkdir(tf)

    for image_class in classes:
        folder = os.path.join(tf, image_class)
        os.mkdir(folder)
        for image_name in classes[image_class]:
            source_path = os.path.join(dataset_folder,"images", image_name)
            dest_path = os.path.join(folder, image_name)
            shutil.copyfile(source_path, dest_path)

    return tf

def define_datasets(folder_name):
    dataset = torchvision.datasets.ImageFolder(root=folder_name, transform=transform)
    label_dict = dataset.class_to_idx
    train_len = int(0.85 * len(dataset))
    test_len = len(dataset) - train_len
    lengths = [train_len, test_len]
    train_set, test_set = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(0))
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader, label_dict

def change_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")
    return device

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.convolutional = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding = 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 64, 3, padding = 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2),

#             nn.Conv2d(64, 128, 3, padding = 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 128, 3, padding = 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2),

#             nn.Conv2d(128, 256, 3, padding = 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 256, 3, padding = 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2),
            
#         )
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(65536, 1000),
#         )
    
#     def forward(self, x):
#         x = self.convolutional(x)
#         # print(np.shape(x))
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def train(model, train_dl, test_dl, device = None, models_folder = "models", rate = 0.005, batch_size = 16, epochs = 10):
    start_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate, momentum = 0.9, weight_decay = 0.005)
    bar_size = 37

    def train_loop():
        size = len(train_dl.dataset)
        model.train()

        for batch, (X, y) in enumerate(train_dl):
            pred = model(X.to(device))
            loss = loss_fn(pred, y.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 50 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]                ")
            
            bar_progress = int(bar_size*((batch + 1) * len(X))/size)
            print(f"{'-'*bar_progress + '>'}{' '*((bar_size-bar_progress)-1)}|", end = '\r')
        print(f"{'-'*bar_size}    ")

    def test_loop():
        print("Calculating Testing Accuracy...", end = '\r')
        model.eval()
        size = len(test_dl.dataset)
        num_batches = len(test_dl)
        test_loss, correct = 0,0
        with torch.no_grad():
            for X, y in test_dl:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error:    Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return f"{(100*correct):>0.1f}"
                
    ETC = "?"
    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}   Training done at: {ETC}\n{'-'*bar_size}        ")
        train_loop()
        acc = test_loop()
        torch.save(model, os.path.join(models_folder, f"model{t+1}__{acc}.pth"))
        time_per_epoch = (time.time()-start_time)/(t+1)
        ETC = int((start_time + epochs * time_per_epoch) -6*3600)
        ETC = f"{time.strftime('%H:%M:%S', time.gmtime(ETC))}"


        


if __name__ == "__main__":

    folder_name = make_prep_folder("surrogate_dataset_xl/surrogate")
    device = change_device()
    train_dl, test_dl, label_dict = define_datasets(folder_name)


    model = VGG16().to(device)

    # model = torch.load('2.6.18.48/model23__17.5.pth').to(device)

    ####### model = models.vgg16().to(device)
    # print(model)
    models_folder = input("name of models folder: \n")
    if os.path.exists(models_folder):
        shutil.rmtree(models_folder)
    os.mkdir(models_folder)

    train(model, train_dl, test_dl, device, rate = 0.001, epochs = 25, models_folder = models_folder)
    
    
    f = open("model_class_assignments.txt", "w")

    lines = []
    for pair in label_dict:
        line = f"{label_dict[pair]}: {pair}"
        lines.append(line)
    f.writelines(lines)
    f.close()

    print("\nModel Training Complete.\n")
    # print(label_dict)

    # model_copy = torch.load("model.pth")