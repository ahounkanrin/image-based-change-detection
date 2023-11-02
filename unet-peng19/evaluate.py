import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
import torchinfo
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io, transform
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import jaccard_score, accuracy_score, precision_score, recall_score, f1_score


seed = 9999
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-3
# DATA_DIR = "/home/anicet/Datasets/ChangeDetectionDataset_Lebedev/Real/subset"
DATA_DIR = "/scratch/hnkmah001/Datasets/ChangeDetectionDataset_Lebedev/Real/subset"
batch_size = 16
num_epochs = 100
epsilon = 1e-20
class_weights = [0.1, 0.9]

class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode="residual"):
        super(ConvUnit, self).__init__()
        self.mode = mode
        self.conv1_unit = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                    kernel_size=kernel_size, padding="same")
        self.bn1_unit = nn.BatchNorm2d(num_features=out_channels)
        self.conv2_unit = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                    kernel_size=kernel_size, padding="same")
        self.bn2_unit = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input_tensor):
        x0 = F.selu(self.conv1_unit(input_tensor))
        x = self.bn1_unit(x0)
        x = F.selu(self.conv2_unit(x))
        if self.mode == "residual":
            return x0 + self.bn2_unit(x)
        else:
            return self.bn2_unit(x)
        

class UnetCD(nn.Module):
    def __init__(self, num_class=1, deep_supervision=False):
        super(UnetCD, self).__init__()
        nb_filters = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.conv11 = ConvUnit(in_channels=6, out_channels=nb_filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv21 = ConvUnit(in_channels=nb_filters[0], out_channels=nb_filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up12 = nn.ConvTranspose2d(in_channels=nb_filters[1], out_channels=nb_filters[0],
                                       kernel_size=2, stride=2)
        self.conv12 = ConvUnit(in_channels=2*nb_filters[0], out_channels=nb_filters[0])
        self.conv31 = ConvUnit(in_channels=nb_filters[1], out_channels=nb_filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up22 = nn.ConvTranspose2d(in_channels=nb_filters[2], out_channels=nb_filters[1],
                                       kernel_size=2, stride=2)
        self.conv22 = ConvUnit(in_channels=2*nb_filters[1], out_channels=nb_filters[1])
        self.up13 = nn.ConvTranspose2d(in_channels=nb_filters[1], out_channels=nb_filters[0],
                                       kernel_size=2, stride=2)
        self.conv13 = ConvUnit(in_channels=3*nb_filters[0], out_channels=nb_filters[0])
        self.conv41 = ConvUnit(in_channels=nb_filters[2], out_channels=nb_filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up32 = nn.ConvTranspose2d(in_channels=nb_filters[3], out_channels=nb_filters[2],
                                       kernel_size=2, stride=2)
        self.conv32 = ConvUnit(in_channels=2*nb_filters[2], out_channels=nb_filters[2])
        self.up23 = nn.ConvTranspose2d(in_channels=nb_filters[2], out_channels=nb_filters[1],
                                       kernel_size=2, stride=2)
        self.conv23 = ConvUnit(in_channels=3*nb_filters[1], out_channels=nb_filters[1])
        self.up14 = nn.ConvTranspose2d(in_channels=nb_filters[1], out_channels=nb_filters[0],
                                       kernel_size=2, stride=2)
        self.conv14 = ConvUnit(in_channels=4*nb_filters[0], out_channels=nb_filters[0])
        self.conv51 = ConvUnit(in_channels=nb_filters[3], out_channels=nb_filters[4])
        self.up42 = nn.ConvTranspose2d(in_channels=nb_filters[4], out_channels=nb_filters[3], 
                                       kernel_size=2, stride=2)
        self.conv42 = ConvUnit(in_channels=2*nb_filters[3], out_channels=nb_filters[3])
        self.up33 = nn.ConvTranspose2d(in_channels=nb_filters[3], out_channels=nb_filters[2], 
                                       kernel_size=2, stride=2)
        self.conv33 = ConvUnit(in_channels=3*nb_filters[2], out_channels=nb_filters[2])
        self.up24 = nn.ConvTranspose2d(in_channels=nb_filters[2], out_channels=nb_filters[1],
                                       kernel_size=2, stride=2)
        self.conv24 = ConvUnit(in_channels=4*nb_filters[1], out_channels=nb_filters[1])
        self.up15 = nn.ConvTranspose2d(in_channels=nb_filters[1], out_channels=nb_filters[0], 
                                       kernel_size=2, stride=2)
        self.conv15 = ConvUnit(in_channels=5*nb_filters[0], out_channels=nb_filters[0])
        self.output1 = nn.Conv2d(in_channels=nb_filters[0], out_channels=num_class, kernel_size=1, padding="same")
        self.output2 = nn.Conv2d(in_channels=nb_filters[0], out_channels=num_class, kernel_size=1, padding="same")
        self.output3 = nn.Conv2d(in_channels=nb_filters[0], out_channels=num_class, kernel_size=1, padding="same")
        self.output4 = nn.Conv2d(in_channels=nb_filters[0], out_channels=num_class, kernel_size=1, padding="same")
        self.output5 = nn.Conv2d(in_channels=4*nb_filters[0], out_channels=num_class, kernel_size=1, padding="same")


    def forward(self, inputs):
        x11 = self.conv11(inputs)
        p1 = self.pool1(x11)
        x21 = self.conv21(p1)
        p2 = self.pool2(x21)
        u12 = self.up12(x21)
        x12 = self.conv12(torch.cat([u12, x11], dim=1))
        x31 = self.conv31(p2)
        p3 = self.pool3(x31)
        u22 = self.up22(x31)
        x22 = self.conv22(torch.cat([u22, x21], dim=1))
        u13 = self.up13(x22)
        x13 = self.conv13(torch.cat([u13, x11, x12], dim=1))
        x41 = self.conv41(p3)
        p4 = self.pool4(x41)
        u32 = self.up32(x41)
        x32 = self.conv32(torch.cat([u32, x31], dim=1))
        u23 = self.up23(x32)
        x23 = self.conv23(torch.cat([u23, x21, x22], dim=1))
        u14 = self.up14(x23)
        x14 = self.conv14(torch.cat([u14, x11, x12, x13], dim=1))
        x51 = self.conv51(p4)
        u42 = self.up42(x51)
        x42 = self.conv42(torch.cat([u42, x41], dim=1))
        u33 = self.up33(x42)
        x33 = self.conv33(torch.cat([u33, x31, x32], dim=1))
        u24 = self.up24(x33)
        x24 = self.conv24(torch.cat([u24, x21, x22, x23], dim=1))
        u15 = self.up15(x24)
        x15 = self.conv15(torch.cat([u15, x11, x12, x13, x14], dim=1))
        out1 = F.sigmoid(self.output1(x12))
        out2 = F.sigmoid(self.output2(x13))
        out3 = F.sigmoid(self.output3(x14))
        out4 = F.sigmoid(self.output4(x15))
        out5 = F.sigmoid(self.output5(torch.cat([x12, x13, x14, x15], dim=1)))

        if self.deep_supervision:
            return [out1, out2, out3, out4, out5]
        else:
            return out5
        
def dice_loss(y_true, y_pred, weight=1.):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_true) + weight * torch.sum(y_pred)
    return 1. - ((2. * intersection + epsilon) / (union + epsilon))

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    bce_weights = class_weights[1] * y_true + class_weights[0] * (1. - y_true)
    bce = (y_true * torch.log(y_pred + epsilon)) + ((1. - y_true) * torch.log(1. - y_pred + epsilon))
    bce = - bce_weights * bce
    bce = torch.mean(bce)
    return bce + 0.5 * dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred):    
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    y_pred = y_pred.round().int()
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_true) +  torch.sum(y_pred)
    return (intersection + epsilon) / (union + epsilon)

class WeightedDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_true, y_pred):
        return weighted_bce_dice_loss(y_true, y_pred)

class ChangeDetectionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        img2_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 2])
        mask_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])

        img1 = io.imread(img1_path)
        img2 = io.imread(img2_path)
        mask = io.imread(mask_path)
        sample = (img1, img2, mask)

        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor:
    def __call__(self, sample):
        img1, img2, mask = sample
        img1 = img1.transpose((2, 0, 1))
        img2 = img2.transpose((2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        return (torch.from_numpy(img1/255.).float(), 
                torch.from_numpy(img2/255.).float(), 
                torch.from_numpy(mask/255.).int())

class ConcatFusion:
    def __call__(self, sample):
        img1, img2, mask = sample
        return torch.cat([img1, img2], axis=0), mask



test_transforms = transforms.Compose([ToTensor(), ConcatFusion()])       

test_dataset = ChangeDetectionDataset(csv_file="./data/test.csv", root_dir=DATA_DIR,
                                      transform=test_transforms)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=8)



model = UnetCD().to(device)
# torchinfo.summary(model, input_size=(1, 6, 256, 256))

criterion = WeightedDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#metric = BinaryJaccardIndex()

model_path = "./model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

test_loss = 0.
test_iou = 0.
test_accuracy = 0.
test_precision = 0.
test_recall = 0.
test_f1_score = 0.

for test_images, test_masks in tqdm(test_dataloader,  desc="testing"):
    with torch.no_grad():
        test_images = test_images.to(device)
        test_masks = test_masks.to(device)

        test_preds = model(test_images)
        test_loss += criterion(test_masks, test_preds).item()
        
        test_iou += jaccard_score(test_preds.view(-1).round().int().cpu(), test_masks.view(-1).cpu())
        test_accuracy += accuracy_score(test_preds.view(-1).round().int().cpu(), test_masks.view(-1).cpu())
        test_precision += precision_score(test_preds.view(-1).round().int().cpu(), test_masks.view(-1).cpu())
        test_recall += recall_score(test_preds.view(-1).round().int().cpu(), test_masks.view(-1).cpu())
        test_f1_score += f1_score(test_preds.view(-1).round().int().cpu(), test_masks.view(-1).cpu())
        # break

test_loss /= len(test_dataloader)
test_iou /= len(test_dataloader)
test_accuracy /= len(test_dataloader)
test_precision /= len(test_dataloader)
test_recall /= len(test_dataloader)
test_f1_score /= len(test_dataloader)

print(f"accuracy = {test_accuracy:.6f}, IoU = {test_iou:.4f}, precision = {test_precision:.4f}, recall = {test_recall:.6f}, f1_score = {test_f1_score:.4f}")




