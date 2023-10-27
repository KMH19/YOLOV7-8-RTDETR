# Cross implementation of YOLO v7, v8 and RT-DETR

Models where mainly trained on RT-DETR in this project, hence the use of YOLO v7 and v8 might not be optimized. 


## Usage
1. Clone the following yolov7 repository and combine with this repository. :)
https://github.com/WongKinYiu/yolov7

2. Download the YOLO optimized dataset from https://www.kaggle.com/datasets/kaspermaarschalk/fscone and extract into the workspace.

3. For YOLO v8 and RT-DETR modules
```
pip install ultralytics
```


## Additional information

The dataset is courtesy of:
https://www.fsoco-dataset.com/

Converted to YOLO txt format and splitted into train/val: 
https://www.kaggle.com/datasets/kaspermaarschalk/fscone

Kaggle notbook for training is included in kaggle_training.ipynb
and cloud config is available at: https://www.kaggle.com/datasets/kaspermaarschalk/kg-conf-new