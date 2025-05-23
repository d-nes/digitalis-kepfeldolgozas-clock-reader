import cv2
import numpy as np
from collections import defaultdict

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
edges = cv2.Canny(blur, 30, 100, apertureSize=3)
# cv2.imshow('Edges', edges)

all_lines = []
lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180, threshold=35,
    minLineLength=int(radius * 0.1),
    maxLineGap=10
)
if lines is not None:
    print(f"Detected {len(lines)} lines.")
    for idx, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        dist1 = np.hypot(x1 - center[0], y1 - center[1])
        dist2 = np.hypot(x2 - center[0], y2 - center[1])
        length = np.hypot(x2 - x1, y2 - y1)
        print(f"Line {idx}: ({x1},{y1})-({x2},{y2}), length={length:.1f}, dist1={dist1:.1f}, dist2={dist2:.1f}")
        if dist1 < radius * 0.3 or dist2 < radius * 0.3:
            all_lines.append((x1, y1, x2, y2, length))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print(f"  -> [kept]")
        else:
            print(f"  -> [discarded]")
else:
    print("No lines detected.")

print(f"Total kept hand candidates: {len(all_lines)}")

hand_infos = []
for (x1, y1, x2, y2, length) in all_lines:
    # Choose the endpoint farther from the center as the tip
    d1 = np.hypot(x1 - center[0], y1 - center[1])
    d2 = np.hypot(x2 - center[0], y2 - center[1])
    tip = (x1, y1) if d1 > d2 else (x2, y2)
    # Angle: 0 is up, increases clockwise
    dx = tip[0] - center[0]
    dy = center[1] - tip[1]
    angle = np.degrees(np.arctan2(dx, dy)) % 360
    hand_infos.append({'angle': angle, 'length': length, 'tip': tip})
    print(f"Hand candidate: angle={angle:.1f}, length={length:.1f}")

def cluster_lines_by_angle(hand_infos, angle_thresh=10):
    clusters = []
    used = [False] * len(hand_infos)
    for i, h in enumerate(hand_infos):
        if used[i]:
            continue
        cluster = [h]
        used[i] = True
        for j, h2 in enumerate(hand_infos):
            if not used[j]:
                diff = abs(h['angle'] - h2['angle'])
                diff = min(diff, 360 - diff)
                if diff < angle_thresh:
                    cluster.append(h2)
                    used[j] = True
        clusters.append(cluster)
    return clusters

clusters = cluster_lines_by_angle(hand_infos, angle_thresh=10)
print(f"Found {len(clusters)} clusters.")

# For each cluster, compute average angle and total length
agg_hands = []
for idx, cluster in enumerate(clusters):
    avg_angle = np.mean([h['angle'] for h in cluster])
    total_length = np.sum([h['length'] for h in cluster])
    agg_hands.append({'angle': avg_angle, 'length': total_length, 'count': len(cluster)})
    print(f"Cluster {idx}: avg_angle={avg_angle:.1f}, total_length={total_length:.1f}, count={len(cluster)}")

# Sort by total length descending (minute > hour > second)
agg_hands = sorted(agg_hands, key=lambda h: h['length'], reverse=False)
# If 4 clusters, drop the shortest
if len(agg_hands) == 4:
    min_length = min(h['length'] for h in agg_hands)
    agg_hands = [h for h in agg_hands if h['length'] > min_length + 1e-3]  # add small epsilon for float safety
    print("Dropped shortest cluster.")

# Compute angle deviation for each cluster
for h in agg_hands:
    h['angle_std'] = 0
for idx, cluster in enumerate(clusters):
    if len(cluster) > 1:
        std = np.std([h['angle'] for h in cluster])
        # Find the corresponding agg_hand by avg_angle
        for agg in agg_hands:
            if abs(agg['angle'] - np.mean([h['angle'] for h in cluster])) < 1e-3:
                agg['angle_std'] = std

# Assign hands:
# - Second hand: smallest angle_std
# - Minute hand: longest (excluding second hand)
# - Hour hand: remaining

agg_hands = sorted(agg_hands, key=lambda h: h['angle_std'])
second = agg_hands[0]
rest = [h for h in agg_hands if h != second]
minute = max(rest, key=lambda h: h['length'])
hour = min(rest, key=lambda h: h['length'])

print(f"Minute hand angle: {minute['angle']:.1f}")
print(f"Hour hand angle: {hour['angle']:.1f}")
print(f"Second hand angle: {second['angle']:.1f}")

def angle_to_time(angle, divisions):
    return (angle / 360) * divisions

minute_value = angle_to_time(minute['angle'], 60)
hour_value = angle_to_time(hour['angle'], 12)
print(f"Estimated time: {int(hour_value)%12:02d}:{int(minute_value):02d}")

# Show result
cv2.imshow('Clock Reader', img)
cv2.waitKey(0)
cv2.destroyAllWindows()