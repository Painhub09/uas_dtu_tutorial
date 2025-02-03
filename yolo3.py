from ultralytics import YOLO
import cv2 as cv
model=YOLO('c:/Users/sahal/OneDrive/Desktop/uas_dtu/photos/best.pt')


results=model.predict('photos/mangplant.jpg', save=True)
results1=model.predict('photos/mangplant(back).jpg', save=True)


image=cv.imread('photos/mangplant(back).jpg')
imagewidth= 640


def extract(results):
    bounding_boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2= box.xyxy[0]
        confidence =box.conf[0]
        class_id =int(box.cls[0])
        object_name ='yellow fruit' if class_id == 1 else 'leaves'
        bounding_boxes.append({
            'x1': float(x1),
            'y1': float(y1),
            'x2': float(x2),
            'y2': float(y2),
            'confidence': float(confidence),
            'class_id': class_id,
            'object_identification': object_name
        })
    return bounding_boxes

bounding_boxes = extract(results)
bounding_boxes1 = extract(results1)


bounding_boxes_leaf = [box for box in bounding_boxes if box['class_id'] == 3]
bounding_boxes_fruit = [box for box in bounding_boxes if box['class_id'] == 1]

bounding_boxes1_leaf = [box for box in bounding_boxes1 if box['class_id'] == 3]
bounding_boxes1_fruit = [box for box in bounding_boxes1 if box['class_id'] == 1]


sorted_bounding_boxes_leaf_ascending = sorted(bounding_boxes_leaf, key=lambda box: box['x1'])
sorted_bounding_boxes1_descending = sorted(bounding_boxes1_leaf, key=lambda box: box['x1'], reverse=True)


def compute_iou(box1, box2,i):

    boxA=sorted_bounding_boxes_leaf_ascending[i]
    boxB=sorted_bounding_boxes1_descending[i]

    # box1['x1']=box1['x1']-boxA['x1']
    # box1['x2']=box1['x2']-boxA['x1']
    # box2['x1']=box2['x1']-boxB['x1']
    # box2['x2']=box2['x2']-boxB['x1']

    # box1['y1']=box1['y1']-boxA['y1']
    # box1['y2']=box1['y2']-boxA['y1']
    # box2['y1']=box2['y1']-boxB['y1']
    # box2['y2']=box2['y2']-boxB['y1']


    x_left= max(box1['x1'], box2['x1'])
    y_top =max(box1['y1'], box2['y1'])
    x_right =min(box1['x2'], box2['x2'])
    y_bottom= min(box1['y2'], box2['y2'])

    intersectionarea =max(0, x_right - x_left)* max(0, y_bottom - y_top)
    area1=(box1['x2'] - box1['x1'])* (box1['y2'] - box1['y1'])
    area2 =(box2['x2'] - box2['x1'])*(box2['y2'] - box2['y1'])
    
    union_area = area1 + area2 - intersectionarea
    return intersectionarea / union_area if union_area > 0 else 0


thres = 0.1
matched_fruits = 0


for i in range(min(len(sorted_bounding_boxes_leaf_ascending), len(sorted_bounding_boxes1_descending))):
    box1= sorted_bounding_boxes_leaf_ascending[i]
    box2=sorted_bounding_boxes1_descending[i]

   
    fruits1 = [box for box in bounding_boxes_fruit if box1['x1'] <= box['x1'] <= box1['x2']]
    fruits2 = [box for box in bounding_boxes1_fruit if box2['x1'] <= box['x1'] <= box2['x2']]

    for fruit1 in fruits1:
        for fruit2 in fruits2:
            mirrored_box = {
                'x1': imagewidth- fruit2['x2'],
                'y1': fruit2['y1'],
                'x2': imagewidth - fruit2['x1'],
                'y2': fruit2['y2']
            }
            if compute_iou(fruit1, mirrored_box,i) >thres:
                matched_fruits += 1

print('matched fruit ',matched_fruits)
total_fruit_count=len(bounding_boxes_fruit)+len(bounding_boxes1_fruit)-matched_fruits

print("Total Fruits (Unique Count):", total_fruit_count)

