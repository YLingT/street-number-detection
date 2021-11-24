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
Download yolov5m pretrain weight: https://github.com/ultralytics/yolov5/releases, put it under weights folder.  

### 1.  Data preparing
create `snd.yaml` in `./data`, 
```
train: data/snd/train  # 33402 images
val: data/snd/test  # no validation data
# number of classes
nc: 10
# class names
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
```
The project structure are as follows:
```
street-number-detection
|── data
  |── snd.yaml
  |── snd
    |── train
      |── 1.png
      |── 1.txt
      |── 2.png
      |── 2.txt...
    |── test
      |── 117.png
      |── 162.png...
|── utils
  |── dataset.py
  |── loss.py ...
|── models
  |── yolo.py ...
|── weights
  |── yolov5m.pt
|── general.py
|── train.py
|── test.py
```
### 2.  Training
Parameter setting:
```
epoch              50
batch size         16
learning rate      1E-2
criterion          FocalLoss
optimizer          SGD (or Adam)
lr scheduler       LambdaLR (or ReduceLROnPlateau)
```
Run code:
```
python train.py --img 320 --batch 16 --epochs 50 --data snd.yaml --weights yolov5m.pt
```
### 3.  Testing
Download trained weight: [best.pt](), put it under weights folder.
Test and generate answer.json:
```
python test.py --source data/snd/test/ --weights weights/best.pt --save-txt
```
Architecture in answer.json:  
bbox = [left, top, width, height]
```
[
    {
        "image_id": 117,
        "score": 0.74707,
        "category_id": 3,
        "bbox": [
            41.999961,
            8.999991999999999,
            12.00005,
            24.999988000000002
        ]
    }, ...
```

### 4.  Result analysis
|   Epoch  |  Optimizer  |  lr scheduler|   Accuracy   |
|   :---:  |    :---:    |     :---:    |    :---:     |
|     50   |     Adam    |  ReduceLROnPlateau    |    0.4098    |
|     **50**   |     **SGD**     |  LambdaLR    |    **0.4147**    |
|     100  |     SGD     |  LambdaLR    |    0.4147    |
|     150  |     SGD     |  LambdaLR    |    0.4134    |
|     300  |     SGD     |  LambdaLR    |    0.3863    |

### 5. Reference
- [YOLOv5](https://github.com/ultralytics/yolov5)

