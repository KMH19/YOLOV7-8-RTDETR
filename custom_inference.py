import os
import GPUtil
import torch
import cv2 as cv
from pathlib import Path
import pandas as pd
from ultralytics import YOLO, RTDETR
from models.yolo import Model
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device

RTDETR_MAP_COCO = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

RTDETR_MAP_FS = {
    0: 'blue_cone',
    1: 'yellow_cone',
    2: 'large_orange_cone',
    3: 'orange_cone',
    4: 'unknown_cone'
}

def print_gpu_memory_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU ID: {gpu.id}")
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Driver Version: {gpu.driver}")
        print(f"GPU Memory Free: {gpu.memoryFree}MB")
        print(f"GPU Memory Used: {gpu.memoryUsed}MB")
        print(f"GPU Memory Total: {gpu.memoryTotal}MB")
        print("------")


dependencies = ['torch', 'yaml']
check_requirements(Path("yolov7/").parent / 'requirements.txt', exclude=('pycocotools', 'thop'))
set_logging()


def load_model(model_name, model_path):
    """
    Load a YOLO or RT-DETR model.
    
    Parameters:
    - model_name (str): One of 'yolov7', 'yolov8', or 'rtdetr'.
    - model_path (str): Path to the pretrained model file.
    
    Returns:
    - model: The loaded model.
    """
    if model_name == 'yolov7':
        return custom(path_or_model=model_path)
    elif model_name == 'yolov8':
        return YOLO(model_path)
    elif model_name == 'rtdetr':
        return RTDETR(model_path)
    else:
        raise ValueError("Unknown model name. Use 'yolov7', 'yolov8', or 'rtdetr'.")



def process_generic(frame, model, model_name, conf_thresh=0.5):
    """Process a single frame and return the processed frame."""
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if model_name == 'yolov7':
        results = model(frame_rgb)
        df_prediction = results.pandas().xyxy[0]
    elif model_name == 'yolov8':
        ultralytics_results = model(frame_rgb, verbose=False)[0]
        data = []
        for box, label, conf in zip(ultralytics_results.boxes.xyxy, ultralytics_results.boxes.cls, ultralytics_results.boxes.conf):
            data.append({
                'xmin': box[0],
                'ymin': box[1],
                'xmax': box[2],
                'ymax': box[3],
                'confidence': conf,
                'name': ultralytics_results.names[int(label)]
            })
        df_prediction = pd.DataFrame(data)
    elif model_name == 'rtdetr':
        ultralytics_results = model(frame_rgb, verbose=False)[0]
        data = []
        for box, label, conf in zip(ultralytics_results.boxes.xyxy, ultralytics_results.boxes.cls, ultralytics_results.boxes.conf):
            data.append({
                'xmin': box[0],
                'ymin': box[1],
                'xmax': box[2],
                'ymax': box[3],
                'confidence': conf,
                'name': RTDETR_MAP_FS[int(label)]
            })
        df_prediction = pd.DataFrame(data)

        # Check if 'confidence' column exists in df_prediction
        if 'confidence' in df_prediction.columns:
            df_prediction = df_prediction[df_prediction['confidence'] >= conf_thresh]
        else:
            print("Warning: Confidence values not found in predictions for rtdetr model. Proceeding without filtering.")
            return draw_bbox(frame, df_prediction)

    else:
        raise ValueError("Unknown model name. Use 'yolov7', 'yolov8', or 'rtdetr'.")
    
    return draw_bbox(frame, df_prediction)

def process_iou_frame(frame, model, model_name, conf_thresh=0.5):
    """Process a single frame and return the processed frame as DataFrame."""
    #frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_rgb = frame
    data = []
    
    if model_name == 'yolov7':
        results = model(frame_rgb)
        df_prediction = results.pandas().xyxy[0]
    elif model_name == 'yolov8':
        ultralytics_results = model(frame_rgb, verbose=False)[0]
        for box, label, conf in zip(ultralytics_results.boxes.xyxy, ultralytics_results.boxes.cls, ultralytics_results.boxes.conf):
            data.append({
                'xmin': box[0],
                'ymin': box[1],
                'xmax': box[2],
                'ymax': box[3],
                'confidence': conf,
                'name': ultralytics_results.names[int(label)]
            })
        df_prediction = pd.DataFrame(data)
    elif model_name == 'rtdetr':
        ultralytics_results = model.predict(frame_rgb, verbose=False)[0]
        for box, label, conf in zip(ultralytics_results.boxes.xyxy, ultralytics_results.boxes.cls, ultralytics_results.boxes.conf):
            data.append({
                'xmin': box[0],
                'ymin': box[1],
                'xmax': box[2],
                'ymax': box[3],
                'confidence': conf,
                'name': RTDETR_MAP_FS[int(label)]
            })
        df_prediction = pd.DataFrame(data)
    else:
        raise ValueError("Unknown model name. Use 'yolov7', 'yolov8', or 'rtdetr'.")
    
    # Check if 'confidence' column exists in df_prediction
    if 'confidence' in df_prediction.columns:
        df_prediction = df_prediction[df_prediction['confidence'] >= conf_thresh]
    else:
        # Create an empty DataFrame with expected columns
        df_prediction = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name'])
    
    return df_prediction


def custom(path_or_model='coneslayer.pt', autoshape=True):
    """custom mode

    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
    return hub_model.to(device)

def draw_bbox(frame, df_prediction):
    """Draw bounding boxes on a frame based on the predictions."""
    for _, row in df_prediction.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        confidence = row['confidence']
        classification = row['name'] if 'name' in df_prediction.columns else 'Unknown'
        label = f"{classification} {confidence:.2f}"
        cv.putText(frame, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # blue text
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # green rectangle
    return frame

def show_img(frame, window_name="Processed Image", scale=0.5):
    """Display an image or video frame."""
    
    # Scale the frame
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    scaled_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    
    cv.imshow(window_name, scaled_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()    
    
def process_image(img_path, model, model_name, conf_thresh):
    """Process and display an image."""
    img = cv.imread(img_path, cv.IMREAD_COLOR)  # BGR
    processed_img = process_generic(img, model, model_name, conf_thresh)
    show_img(processed_img)


def process_video(video_path, model, model_name, conf_thresh=0.5, output_path='output_video.mp4'):
    """Process and display a video and save the processed video."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_generic(frame, model, model_name, conf_thresh)
        
        # Write the processed frame to the output video
        out.write(processed_frame)
        
        cv.imshow('Processed Video', processed_frame)
        if cv.waitKey(1) == ord('q'):
            break

    # Release everything
    cap.release()
    out.release()  # Release the VideoWriter
    cv.destroyAllWindows()


def process_videostream(model, model_name, conf_thresh=0.5):
    """Process and display a webcam feed."""
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set the resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_generic(frame, model, model_name, conf_thresh)
        #print_gpu_memory_usage()
        cv.imshow('Processed Webcam Feed', processed_frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    
def bbox_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    
    Parameters:
    - boxA, boxB: bounding boxes in the format [x1, y1, x2, y2].
    
    Returns:
    - iou: IoU value.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IoU
    iou = inter_area / float(boxA_area + boxB_area - inter_area)

    return iou

# #? Test the IoU function
# boxA = [50, 50, 150, 150]
# boxB = [100, 100, 200, 200]
# iou = bbox_iou(boxA, boxB)
# print(iou)


def yolo_to_xyxy(bbox, img_width, img_height):
    """
    Convert a YOLO formatted bounding box (center x, center y, width, height) 
    to the format (x1, y1, x2, y2).
    
    Parameters:
    - bbox: bounding box in YOLO format [center x, center y, width, height].
    - img_width: width of the image.
    - img_height: height of the image.
    
    Returns:
    - bbox_xyxy: bounding box in the format [x1, y1, x2, y2].
    """
    cx, cy, w, h = bbox
    cx *= img_width
    cy *= img_height
    w *= img_width
    h *= img_height
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return [x1, y1, x2, y2]

# #? Test the conversion function
# bbox_yolo = [0.5, 0.5, 0.5, 0.5]  # Assume center of the image and half its size
# img_width, img_height = 400, 400
# bbox_xyxy = yolo_to_xyxy(bbox_yolo, img_width, img_height)
# prit(bbox_xyxy)

def load_ground_truth(txt_path, img_width, img_height):
    """
    Load ground truth bounding boxes from a YOLO formatted txt file 
    and convert to the format (x1, y1, x2, y2).
    
    Parameters:
    - txt_path: path to the YOLO formatted txt file.
    - img_width: width of the image.
    - img_height: height of the image.
    
    Returns:
    - ground_truths: list of bounding boxes in the format [x1, y1, x2, y2].
    """
    ground_truths = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            _, cx, cy, w, h = map(float, line.strip().split())
            bbox = yolo_to_xyxy([cx, cy, w, h], img_width, img_height)
            ground_truths.append(bbox)
    return ground_truths

# #? Test the function
# sample_txt_path = Path("/fsoco_dataset/images/val/sample.txt")
# if sample_txt_path.exists():
#     ground_truths = load_ground_truth(sample_txt_path, 1280, 720)
# else:
#     ground_truths = "Sample txt path doesn't exist."
# print(ground_truths)

def compute_iou_for_predictions(img_folder_path, label_folder_path, model, model_name):
    """
    Compute the IoU for each predicted bounding box with its corresponding ground truth.
    
    Parameters:
    - img_folder_path: path to the folder containing validation images.
    - label_folder_path: path to the folder containing validation labels.
    - model: the trained model.
    - model_name: name of the model ('rtdetr', 'yolov7', etc.).
    
    Returns:
    - iou_values: list of IoU values for all predicted bounding boxes.
    """
    img_folder = Path(img_folder_path)
    label_folder = Path(label_folder_path)
    iou_values = []
    i = 0
    for img_path in img_folder.glob("*.jpg"):
        print("New frame: ", i)
        # Load image
        img = cv.imread(str(img_path), cv.IMREAD_COLOR)  # BGR
        
        # Resize the image
        resize_factor = 640.0 / max(img.shape[0], img.shape[1])
        img_resized = cv.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        
        if DEBUGGING:
            cv.imshow('RSIZ', img_resized)
            cv.waitKey(0)
            cv.destroyAllWindows()

        # Run inference on the image to get predicted bounding boxes
        df_prediction = process_iou_frame(img_resized, model, model_name)

        # Adjust bounding box coordinates to the original image dimensions
        df_prediction[['xmin', 'ymin', 'xmax', 'ymax']] /= resize_factor

        # Extract bounding boxes from the DataFrame
        predicted_bboxes = df_prediction[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

        # Load corresponding ground truth bounding boxes
        txt_path = label_folder / img_path.name.replace('.jpg', '.txt')
        ground_truth_bboxes = load_ground_truth(txt_path, img.shape[1], img.shape[0])

        # For each predicted bounding box, compute IoU with ground truth
        for pred_bbox in predicted_bboxes:
            max_iou = max([bbox_iou(pred_bbox, gt_bbox) for gt_bbox in ground_truth_bboxes])
            iou_values.append(max_iou)
            
        # Draw bounding boxes on the frame
        frame_with_bboxes = draw_bboxes_and_ground_truth(img, df_prediction, ground_truth_bboxes)

        # Optionally display the frame
        if DEBUGGING:
            cv.imshow('Frame with Bounding Boxes', frame_with_bboxes)
            cv.waitKey(0)
            cv.destroyAllWindows()
            
        i += 1

    return iou_values

def draw_bboxes_and_ground_truth(frame, df_prediction, ground_truth_bboxes):
    """Draw both predicted bounding boxes and ground truth on a frame with class, confidence, and IoU."""
    
    # Draw predicted bounding boxes
    for _, row in df_prediction.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        confidence = row['confidence']
        classification = row['name'] if 'name' in df_prediction.columns else 'Unknown'
        
        # For each predicted bbox, find the IoU with the ground truth
        max_iou = 0
        for gt_bbox in ground_truth_bboxes:
            iou = bbox_iou([xmin, ymin, xmax, ymax], gt_bbox)
            max_iou = max(max_iou, iou)
        
        label = f"{classification} {confidence:.2f} IoU: {max_iou:.2f}"
        cv.putText(frame, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # white text
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # green rectangle
    
    # Draw ground truth bounding boxes
    for gt_bbox in ground_truth_bboxes:
        xmin, ymin, xmax, ymax = map(int, gt_bbox)
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # red rectangle
        #cv.putText(frame, "Ground Truth", (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # white text

    return frame


#? # Testing inference ----------------------------------------------------------
DEBUGGING = True
model_name = 'rtdetr'  # or 'yolov7' or 'yolov8'
model_path = r'C:\Users\kmhyt\Documents\GitHub\P7\yolov7\c_20_epoch.pt'  # or 'coneslayer.pt' or 'yolov7.pt' or 'rtdetr-l.pt'

model = load_model(model_name, model_path) # Utilizes the custom fnc

# Compute IoU values for predictions
img_folder_path = "fsoco_yolo/images/val/"
label_folder_path = "fsoco_yolo/labels/val/"

if Path(img_folder_path).exists() and Path(label_folder_path).exists():
    iou_values = compute_iou_for_predictions(img_folder_path, label_folder_path, model, model_name)
    mean_iou = sum(iou_values) / len(iou_values) if iou_values else 0
else:
    mean_iou = "Image or label folder path doesn't exist."

print("mean IoU:", mean_iou)

#? -----------------------------------------------------------------------------
# mean IoU: tensor(0.87147, device='cuda:0') # c_20_epoch.pt
# mean IoU: tensor(0.83553, device='cuda:0') # kaggle_2t4.pt

#? # Normal inference ----------------------------------------------------------
#? process_image("61-wEgVmgvL.jpg", model, model_name, conf_thresh=0.6)
#? process_video('video2.mp4', model, model_name, conf_thresh=0.6)
#? process_videostream(model, model_name, conf_thresh=0.4)
#? -----------------------------------------------------------------------------

