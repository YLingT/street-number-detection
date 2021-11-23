# street-number-detection

This is the code for number detection using yolov5 on pytorch.  
If you want to reproduce the project on colab, check: 

## Enviroment setting and dependencies 
Use pip install or conda install :
```
conda create --name test python=3.7.11
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas==1.1.3
pip install matplotlib==3.4.3
```
And check the version :
```
#Name                        Version
python                       3.7.11
torch                        1.7.0
torchvision                  0.8.1
pandas                       1.1.3
pillow                       8.4.0
matplotlib                   3.4.3
```

## Dataset 
There are 33402 images for training and 13068 images for testing.

## Code 
### 0. Download Project
```
git clone https://github.com/YLingT/street-number-detection
cd street-number-detection
```
The project structure are as follows:
```
street-number-detection
|── utils
  |── dataset.py
  |── loss.py ...
|── models
  |── yolo.py ...
|── general.py
|── train.py
|── test.py
```
### 1.  Training
Parameter setting:
```
epoch              50
batch size         16
learning rate      1E-2
criterion          FocalLoss
optimizer          SGD
lr scheduler       LambdaLR
```
Run code:
