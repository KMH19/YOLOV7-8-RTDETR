import os

def load_ground_truth(txt_path, img_width, img_height):
    ground_truths = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            _, cx, cy, w, h = map(float, line.strip().split())
            bbox = yolo_to_xyxy([cx, cy, w, h], img_width, img_height)
            ground_truths.append(bbox)
    return ground_truths

def yolo_to_xyxy(bbox, img_width, img_height):
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

def count_ground_truth_boxes(label_folder_path):
    total_boxes = 0
    for txt_file in os.listdir(label_folder_path):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(label_folder_path, txt_file)
            ground_truths = load_ground_truth(txt_path, img_width=1280, img_height=720)
            total_boxes += len(ground_truths)
    return total_boxes

def main():
    label_folder_path = r"C:\Users\kmhyt\Documents\GitHub\P7\yolov7\fsoco_yolo\labels\val"  # Replace with your label folder path
    total_boxes = count_ground_truth_boxes(label_folder_path)
    print(f"Total ground truth boxes: {total_boxes}")

if __name__ == "__main__":
    main()