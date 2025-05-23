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
        print(f"Detected clock face center: {center}, radius: {radius}")
        cv2.circle(img, center, radius, (0, 255, 0), 2)
else:
    print("No clock face detected.")
    exit()

# Edge detection
edges = cv2.Canny(blur, 30, 100, apertureSize=3)  # Lowered thresholds for more edges
cv2.imshow('Edges', edges)

# Detect lines (potential hands)
lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180, threshold=60,  # Lowered threshold for more lines
    minLineLength=radius//2, maxLineGap=30
)
if lines is not None:
    print(f"Detected {len(lines)} lines.")
    hand_lines = []
    for idx, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        # Only keep lines that pass near the center (hand candidates)
        dist1 = np.hypot(x1 - center[0], y1 - center[1])
        dist2 = np.hypot(x2 - center[0], y2 - center[1])
        if dist1 < radius * 0.2 or dist2 < radius * 0.2:
            hand_lines.append(line[0])
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print(f"Line {idx}: ({x1},{y1})-({x2},{y2}) [kept]")
        else:
            print(f"Line {idx}: ({x1},{y1})-({x2},{y2}) [discarded]")
    print(f"Kept {len(hand_lines)} hand candidates.")
else:
    print("No lines detected.")

# Show result
cv2.imshow('Clock Reader', img)
cv2.waitKey(0)
cv2.destroyAllWindows()