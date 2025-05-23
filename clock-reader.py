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
# cv2.imshow('Edges', edges)

all_lines = []
for edge_img, label in [(edges, "original")]:
    lines = cv2.HoughLinesP(
        edge_img, 1, np.pi / 180, threshold=35,
        minLineLength=int(radius * 0.1),
        maxLineGap=10
    )
    if lines is not None:
        print(f"[{label}] Detected {len(lines)} lines.")
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            dist1 = np.hypot(x1 - center[0], y1 - center[1])
            dist2 = np.hypot(x2 - center[0], y2 - center[1])
            length = np.hypot(x2 - x1, y2 - y1)
            print(f"[{label}] Line {idx}: ({x1},{y1})-({x2},{y2}), length={length:.1f}, dist1={dist1:.1f}, dist2={dist2:.1f}")
            if dist1 < radius * 0.3 or dist2 < radius * 0.3:
                all_lines.append((x1, y1, x2, y2, length, label))
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0) if label == "dilated" else (0, 0, 255), 1)
                print(f"  -> [kept]")
            else:
                print(f"  -> [discarded]")
    else:
        print(f"[{label}] No lines detected.")

print(f"Total kept hand candidates: {len(all_lines)}")

# Show result
cv2.imshow('Clock Reader', img)
cv2.waitKey(0)
cv2.destroyAllWindows()