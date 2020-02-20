import cv2

def draw_box(image, box, color, thickness=2):
  cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness, cv2.LINE_AA)
