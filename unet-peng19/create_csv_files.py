import os 
import pandas as pd

# DATA_DIR = "/home/anicet/Datasets/ChangeDetectionDataset_Lebedev/Real/subset"
DATA_DIR = "/scratch/hnkmah001/Datasets/ChangeDetectionDataset_Lebedev/Real/subset"

if not os.path.isdir("./data"):
    os.makedirs("./data")
    
train_imgs_A = []
train_imgs_B = []
train_imgs_OUT = []
train_df = pd.DataFrame()

for x in os.listdir(os.path.join(DATA_DIR, "train", "A")):
    if x.endswith(".jpg"):
        train_imgs_A.append(os.path.join("train", "A", x))
        train_imgs_B.append(os.path.join("train", "B", x))
        train_imgs_OUT.append(os.path.join("train", "OUT", x))

train_df["image1"] = train_imgs_A
train_df["image2"] = train_imgs_B
train_df["mask"] = train_imgs_OUT
train_df.to_csv("./data/train.csv", sep=",")


val_imgs_A = []
val_imgs_B = []
val_imgs_OUT = []
val_df = pd.DataFrame()

for x in os.listdir(os.path.join(DATA_DIR, "val", "A")):
    if x.endswith(".jpg"):
        val_imgs_A.append(os.path.join("val", "A", x))
        val_imgs_B.append(os.path.join("val", "B", x))
        val_imgs_OUT.append(os.path.join("val", "OUT", x))

val_df["image1"] = val_imgs_A
val_df["image2"] = val_imgs_B
val_df["mask"] = val_imgs_OUT
val_df.to_csv("./data/val.csv", sep=",")


test_imgs_A = []
test_imgs_B = []
test_imgs_OUT = []
test_df = pd.DataFrame()

for x in os.listdir(os.path.join(DATA_DIR, "test", "A")):
    if x.endswith(".jpg"):
        test_imgs_A.append(os.path.join("test", "A", x))
        test_imgs_B.append(os.path.join("test", "B", x))
        test_imgs_OUT.append(os.path.join("test", "OUT", x))

test_df["image1"] = test_imgs_A
test_df["image2"] = test_imgs_B
test_df["mask"] = test_imgs_OUT
test_df.to_csv("./data/test.csv", sep=",")


