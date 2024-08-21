from ultralytics import YOLO
import os
from IPython.display import display,Image, clear_output 
import cv2


image_path = "test images\healthy.jpg"

model = YOLO("best.pt")
results = model.predict("test images\healthy.jpg")

# Get the first result (assuming single image input)
result = results[0]

# Get the original image
img = cv2.imread(image_path)

# Plot the bounding boxes and labels on the image
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf)
    cls = int(box.cls)
    label = f"{model.names[cls]} {conf:.2f}"
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
                