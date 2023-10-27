from ultralytics import RTDETR, YOLO

def main():
    # # Load an RT-DETR model
    # model = RTDETR('rtdetr-l.yaml')  # build a new model from YAML
    # #model = RTDETR('rtdetr.pt')  # load a pretrained model (recommended for training)
    # # model = RTDETR('rtdetr.yaml').load('rtdetr.pt')  # build from YAML and transfer weights

    # # Train the model
    # results = model.train(data=r'fsoco_yolo\config.yaml', epochs=100, imgsz=640, batch=8)
    
    
        # Load an RT-DETR model
    model = RTDETR('kaggle_2t4.pt')  # build a new model from YAML
    #model = RTDETR('rtdetr.pt')  # load a pretrained model (recommended for training)
    # model = RTDETR('rtdetr.yaml').load('rtdetr.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r'fsoco_yolo\config.yaml', epochs=10, imgsz=640, batch=4)

if __name__ == '__main__':
    main()

