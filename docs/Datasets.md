# Overall datasets
### 8 classes datasets:
- **Cityscapes**: Contains 2,975 training images and 500 testing images.
- **Foggy-Cityscapes**: Includes 2,975 training images and 500 testing images, with **a fog level of 0.02** applied.
### 7 classes dataset:
- **BDD100K**: Comprises 36,728 training images and 5,258 testing images.
### 20 classes dataset:
- **Clipart**: A total of 1,000 images, used for both training and testing.
### 1 classes datasets:
- **KITTI**: Includes 7,481 images, all of which are used for training and testing.
- **Sim10K**: Includes 10,000 images, used entirely for training and testing.

# Prepare image lists
First make dirs for datasets:
```bash
mkdir -p  your_datasets_dir/CityScapes_FoggyCityScapes
mkdir -p  your_datasets_dir/BDD100K_voc
mkdir -p  your_datasets_dir/clipart
mkdir -p  your_datasets_dir/KITTI
mkdir -p  your_datasets_dir/SIM
```
Then clone this repository, and run the following to copy image lists to your `your_datasets_dir`:
```bash
git clone https://github.com/Flashkong/COIN.git

cp -r COIN/datasets/* your_datasets_dir
```
The following is the explanation of the image lists.
> The files surrounded by "" are the lists of images we used in the paper.
```bash
your_datasets_dir
├── BDD100K_voc
│   └── ImageSets
│       └── Main
│           ├── train_no_object.txt     # train images with no objects. 134 images
│           ├── "train_object.txt"      # train images with objects. 36,594 images
│           ├── train.txt               # all train images. 36,728 images
│           ├── val_no_object.txt       # val images with no objects. 25 images
│           ├── "val_object.txt"        # val images with objects. 5,233 images
│           └── val.txt                 # all val images. 5,258 images
├── CityScapes_FoggyCityScapes
│   └── ImageSets
│       └── Main
│           ├── train_city_car.txt      # CityScapes train images that only contains cars. 2,831 images
│           ├── "train_city.txt"        # CityScapes train images. 2,965 images. As the remaining 10 images have no objects
│           ├── "train_foggy_0.02.txt"  # FoggyCityScapes train images with a fog level of 0.02. 2,965 images. As the remaining 10 images have no objects
│           ├── train_foggy.txt         # FoggyCityScapes train images with all levels. 8,895 images. As the remaining 30 images have no objects
│           ├── val_city_car.txt        # CityScapes val images that only contains cars. 478 images. As the remaining 22 images have no objects
│           ├── "val_city.txt"          # CityScapes val images. 492 images. As the remaining 8 images have no objects
│           ├── "val_foggy_0.02.txt"    # FoggyCityScapes val images with a fog level of 0.02. 492 images. As the remaining 8 images have no objects
│           └── val_foggy.txt           # FoggyCityScapes val images with all levels. 1476 images. As the remaining 24 images have no objects
├── clipart
│   └── ImageSets
│       └── Main
│           ├── "all.txt"               # all images. 1,000 images
│           ├── test.txt                # test images. 500 images
│           └── train.txt               # train images. 500 images
├── KITTI
│   └── ImageSets
│       └── Main
│           ├── train.txt               # all 7,481 images
│           └── "train_car.txt"         # Images with cars. 6,684 images
│           ├── train_no_car.txt        # Images with no cars. 25 images
└── SIM
    └── ImageSets
        └── Main
            ├── "train_car.txt"         # Images with cars. 9,975 images
            ├── train_no_car.txt        # Images with no cars. 25 images
            └── train.txt               # all 10,000 images
```

# Prepare images and annotations
We do not hold the copyright to the images and annotations in the datasets, but to avoid the tedium of downloading and processing the data, we are making available our local copy of the data.
### Cityscapes & Foggy-Cityscapes
- Register an account [here](https://www.cityscapes-dataset.com/) and download two files: `leftImg8bit_trainvaltest.zip (11GB)` and `leftImg8bit_trainvaltest_foggy.zip (30GB)`. The first is for Cityscapes, and the later is for Foggy-Cityscapes.
- Extract files and Put all images into `your_datasets_dir/CityScapes_FoggyCityScapes/JPEGImages`
- Download our converted annotations `CityScapes_FoggyCityScapes_Annotations.zip` from [here](https://huggingface.co/Flashkong/COIN/tree/main/datasets), and put all `.xml` files at `your_datasets_dir/CityScapes_FoggyCityScapes/Annotations`

### BDD100K_voc
- Download images `BDD100K_voc_JPEGImages.zip` and annotations `BDD100K_voc_Annotations.zip` from [here](https://huggingface.co/Flashkong/COIN/tree/main/datasets), it removes some unused files. 
- Extract `BDD100K_voc_JPEGImages.zip` and put all images at `your_datasets_dir/BDD100K_voc/JPEGImages`. 
- Extract `BDD100K_voc_Annotations.zip` and put all `.xml` files at `your_datasets_dir/BDD100K_voc/Annotations`
- The official website: [BDD100K](https://bdd-data.berkeley.edu/).

### Clipart
- Download images `clipart_JPEGImages.zip` and annotations `clipart_Annotations.zip` from [here](https://huggingface.co/Flashkong/COIN/tree/main/datasets). 
- Extract `clipart_JPEGImages.zip` and put all images at `your_datasets_dir/clipart/JPEGImages`. 
- Extract `clipart_Annotations.zip` and put all `.xml` files at `your_datasets_dir/clipart/Annotations`
- Or you can download from [original paper](https://github.com/naoto0804/cross-domain-detection/blob/master/datasets/README.md).

### KITTI
- Download images `KITTI_JPEGImages.zip` and annotations `KITTI_Annotations.zip` from [here](https://huggingface.co/Flashkong/COIN/tree/main/datasets). 
- Extract `KITTI_JPEGImages.zip` and put all images at `your_datasets_dir/KITTI/JPEGImages`. 
- Extract `KITTI_Annotations.zip` and put all `.xml` files at `your_datasets_dir/KITTI/Annotations`

### Sim10K
- Download images `SIM_JPEGImages.zip` and annotations `SIM_Annotations.zip` from [here](https://huggingface.co/Flashkong/COIN/tree/main/datasets). 
- Extract `SIM_JPEGImages.zip` and put all images at `your_datasets_dir/SIM/JPEGImages`. 
- Extract `SIM_Annotations.zip` and put all `.xml` files at `your_datasets_dir/SIM/Annotations`
- Or you can download from [official website](https://fcav.engin.umich.edu/projects/driving-in-the-matrix).

### Check md5
Check downloaded files using:
```bash
md5sum *.zip
```
The output should be:
```bash
fccdbe5d0b659377d470b78b16d2a9a1  BDD100K_voc_Annotations.zip
ee91ef5d0001dd80f919c8c339a0bca5  BDD100K_voc_JPEGImages.zip
4fe2995dbd23042609fcc5de3fb57483  CityScapes_FoggyCityScapes_Annotations.zip
5e90115651a18271c4d559bbd38a4553  clipart_Annotations.zip
4aa00918c0ee6ee3534b950d9939ac17  clipart_JPEGImages.zip
20a98ea41c19fd220b39329744f3932a  KITTI_Annotations.zip
18e9d7d460683265ed3dcc0d7f7b032c  KITTI_JPEGImages.zip
55d8b00893768211da806de7e2d8bb08  SIM_Annotations.zip
315b55679d62ce78207c3a128beba801  SIM_JPEGImages.zip
```