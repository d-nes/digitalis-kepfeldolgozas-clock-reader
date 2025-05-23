import cv2
import numpy as np

# Load image
img = cv2.imread('test-images/faliora.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Detect clock face (circle)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                           param1=50, param2=30, minRadius=100, maxRadius=0)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :1]:  # Only use the largest/first circle
        center = (i[0], i[1])
        radius = i[2]
        cv2.circle(img, center, radius, (0, 255, 0), 2)
else:
    print("No clock face detected.")
    exit()

# Edge detection
edges = cv2.Canny(blur, 50, 150, apertureSize=3)

# Detect lines (potential hands)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=radius//2, maxLineGap=20)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Show result
cv2.imshow('Clock Reader', img)
cv2.waitKey(0)
cv2.destroyAllWindows()