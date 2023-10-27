import os
import json
import shutil
from PIL import Image


# Settings
path = "fsoco_bounding_boxes_train"
root_dirs = ["ampera", "amz", "aristurtle", "asurt", "baltic", "bauman", "bme", "dhen", "dtu", "eco", "ecurieaix", "epflrt", "eufs", "ff", "frt", "fsb", "fsbizkaia", "gfr", "iitb", "ka", "kth", "mad", "mms", "msm", "orion", "penn", "prc", "prom", "pwrrt", "racetech", "rennstall", "tuwr", "ugatu", "ugent", "ugr", "ulm", "umsae", "unicamp", "unipg", "uop", "wfm"]
class_names = ["blue_cone", "yellow_cone", "large_orange_cone", "orange_cone", "unknown_cone"]

output_image_dir = "fsoco_yolo/images"
output_label_dir = "fsoco_yolo/labels"

def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    Convert COCO bounding box format [x, y, width, height] to YOLO format [x_center, y_center, width, height]
    """
    x, y, w, h = coco_bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height
    return [x_center, y_center, w, h]

def get_coco_bbox_from_points(points):
    """
    Get COCO-style bounding box [x, y, width, height] from the exterior points.
    """
    x_min, y_min = points['exterior'][0]
    x_max, y_max = points['exterior'][1]
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def convert_annotations(root_dir, img_subdir='img', ann_subdir='ann'):
    img_dir = os.path.join(root_dir, img_subdir)
    ann_dir = os.path.join(root_dir, ann_subdir)
    
    for img_file in os.listdir(img_dir):
        img_name, img_ext = os.path.splitext(img_file)
        if img_ext not in ['.png', '.jpg']:
            continue
        
        # Get the width and height of the image
        with Image.open(os.path.join(img_dir, img_file)) as img:
            img_width, img_height = img.size
        
        # Find the corresponding annotation file
        ann_file = os.path.join(ann_dir, img_name + img_ext + '.json')
        
        # Check if annotation file exists
        if not os.path.exists(ann_file):
            print(f"Annotation for {img_file} not found!")
            continue
        
        # Read the annotation file
        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Extract image dimensions and annotations
        img_width = data['size']['width']
        img_height = data['size']['height']
        annotations = data['objects']

        # Initialize YOLO formatted annotations
        yolo_anns = []

        for ann in annotations:
            class_title = ann['classTitle']
            if class_title in class_names:
                coco_bbox = get_coco_bbox_from_points(ann['points'])
                yolo_bbox = coco_to_yolo_bbox(coco_bbox, img_width, img_height)
                yolo_anns.append([class_names.index(class_title)] + yolo_bbox)

        # Save the YOLO formatted annotations
        yolo_ann_file = os.path.join(output_label_dir, img_name + r'.txt')
        
        #print(os.listdir(output_label_dir))
        
        with open(yolo_ann_file, 'w') as f:
            for ann in yolo_anns:
                f.write(' '.join(map(str, ann)) + '\n')

        # Move the image to the output_image_dir
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(output_image_dir, img_file))

if __name__ == "__main__":
    for dir_name in root_dirs:
        convert_annotations(os.path.join(path, dir_name))
