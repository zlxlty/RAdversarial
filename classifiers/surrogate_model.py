import torchvision
import torch
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
import os
import shutil
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import numpy as np
import torchvision.models as models
import time
import datetime
torch.manual_seed(0)
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
    train_len = int(0.9375 * len(dataset))
    test_len = len(dataset) - train_len
    lengths = [train_len, test_len]
    train_set, test_set = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(0))
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader, label_dict

def change_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    return device

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

def add_to_tracker(tracker_file, output_line):
    file = open(tracker_file, 'a')
    file.write(f"{output_line}\n")
    file.close()


def train(model, train_dl, test_dl, device = None, models_folder = "models", rate = 0.005, batch_size = 16, epochs = 10, tracker_file = "tracker.txt", weight_decay = 0.005):
    start_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate, momentum = 0.9, weight_decay = weight_decay)
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

            if batch % 100 == 0:  #just changed
                loss, current = loss.item(), (batch + 1) * len(X)
                outie = f"training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]                "
                print(outie)
                add_to_tracker(tracker_file, outie)
            progress = ((batch + 1) * len(X))/size
            bar_progress = int(bar_size*progress)
            print(f"{'-'*bar_progress + '>'}{' '*((bar_size-bar_progress)-1)}| {round(100*progress, 1):>3}%", end = '\r')
        end_line = f"{'-'*bar_size}           "
        print(end_line)
        add_to_tracker(tracker_file, end_line)

    def test_loop():
        percent = " "
        print(f"Calculating Testing Accuracy... {percent}", end = '\r')
        model.eval()
        size = len(test_dl.dataset)
        num_batches = len(test_dl)
        test_loss, correct = 0,0
        with torch.no_grad():
            count = 0
            for X, y in test_dl:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                percent = f'{round(100*(count/num_batches), 1)}'
                print(f"Calculating Testing Accuracy... {percent}%", end = '\r')
                count += 1
        test_loss /= num_batches
        correct /= size
        accuracy_line = f"Test Error:    Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n"
        print(accuracy_line)
        add_to_tracker(tracker_file, accuracy_line)
        return f"{(100*correct):>0.1f}"
                
    ETC = "?"
    for t in range(epochs):
        epoch_title = f"Epoch {t+1}/{epochs}   Training done at: {ETC}\n{'-'*bar_size}        "
        print(epoch_title)
        add_to_tracker(tracker_file, epoch_title)
        train_loop()
        acc = test_loop()
        torch.save(model, os.path.join(models_folder, f"new_split_model{t+1}__{acc}.pth"))
        time_per_epoch = (time.time()-start_time)/(t+1)
        ETC = int((start_time + epochs * time_per_epoch) -6*3600)
        ETC = f"{time.strftime('%H:%M:%S (%A)', time.gmtime(ETC))}"


        
if __name__ == "__main__":
    # print("preparing dataset:")
    # folder_name = make_prep_folder("surrogate_dataset_xl/surrogate")
    # train_dl, test_dl, label_dict = define_datasets(folder_name)

    # print("preparing model:")
    device = change_device()
    # model = VGG16().to(device)

    model_source = '2.11.23.55/new_split_model30__30.4.pth'
    model = torch.load(model_source).to(device)

    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    print(state_dict[keys[0]][:5])






    
    # print(model_source)

    # models_folder = input("name of models folder: \n")
    

    # if os.path.exists(models_folder):
    #     shutil.rmtree(models_folder)
    # os.mkdir(models_folder)

    # tracker_file = os.path.join(models_folder, "terminal_output.txt")
    # add_to_tracker(tracker_file, model_source)

    # f = open("model_class_assignments.txt", "w")

    # lines = []
    # for pair in label_dict:
    #     line = f"{label_dict[pair]}: {pair}"
    #     lines.append(line)
    # f.writelines(lines)
    # f.close()

    # train(model, train_dl, test_dl, device, rate = 0.003, epochs = 30, models_folder = models_folder, tracker_file = tracker_file, weight_decay = 0.001)
    # models_folder = os.path.join(models_folder, ".2")
    # if os.path.exists(models_folder):
    #     shutil.rmtree(models_folder)
    # os.mkdir(models_folder)
    # train(model, train_dl, test_dl, device, rate = 0.001, epochs = 50, models_folder = models_folder, tracker_file = tracker_file, weight_decay = 0.002)
    


    print("\nModel Training Complete.\n")
    # print(label_dict)

    # model_copy = torch.load("model.pth")