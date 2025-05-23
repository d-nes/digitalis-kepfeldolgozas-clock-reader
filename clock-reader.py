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
agg_hands = sorted(agg_hands, key=lambda h: h['length'], reverse=True)
if len(agg_hands) >= 2:
    minute = agg_hands[0]
    hour = agg_hands[1]
    print(f"Minute hand angle: {minute['angle']:.1f}")
    print(f"Hour hand angle: {hour['angle']:.1f}")
    if len(agg_hands) > 2:
        second = agg_hands[2]
        print(f"Second hand angle: {second['angle']:.1f}")

        def angle_to_time(angle, divisions):
            return (angle / 360) * divisions

        minute_value = angle_to_time(minute['angle'], 60)
        hour_value = angle_to_time(hour['angle'], 12)
        print(f"Estimated time: {int(hour_value)%12:02d}:{int(minute_value):02d}")
else:
    print("Not enough hands detected after clustering.")

# Merge closest clusters until only 3 remain
def merge_closest_clusters(agg_hands, target_count=3):
    while len(agg_hands) > target_count:
        # Find pair with smallest angle difference
        min_diff = 360
        min_i, min_j = 0, 1
        for i in range(len(agg_hands)):
            for j in range(i+1, len(agg_hands)):
                diff = abs(agg_hands[i]['angle'] - agg_hands[j]['angle'])
                diff = min(diff, 360 - diff)
                if diff < min_diff:
                    min_diff = diff
                    min_i, min_j = i, j
        # Merge clusters i and j
        c1, c2 = agg_hands[min_i], agg_hands[min_j]
        total_length = c1['length'] + c2['length']
        avg_angle = (c1['angle'] * c1['length'] + c2['angle'] * c2['length']) / total_length
        merged = {
            'angle': avg_angle,
            'length': total_length,
            'count': c1['count'] + c2['count']
        }
        # Remove and insert merged
        new_agg = [agg_hands[k] for k in range(len(agg_hands)) if k not in (min_i, min_j)]
        new_agg.append(merged)
        agg_hands = new_agg
    return agg_hands

agg_hands = merge_closest_clusters(agg_hands, target_count=3)
print(f"After merging, {len(agg_hands)} clusters remain.")
for idx, h in enumerate(agg_hands):
    print(f"Cluster {idx}: avg_angle={h['angle']:.1f}, total_length={h['length']:.1f}, count={h['count']}")

agg_hands = sorted(agg_hands, key=lambda h: h['length'], reverse=True)

if len(agg_hands) >= 2:
    # Minute hand: longest
    minute = agg_hands[0]
    # Hour hand: closest angle to minute hand (but not the minute hand itself)
    hour_candidates = agg_hands[1:]
    hour = min(hour_candidates, key=lambda h: min(abs(h['angle'] - minute['angle']), 360 - abs(h['angle'] - minute['angle'])))
    # Second hand: the remaining cluster (if present)
    second = None
    if len(agg_hands) > 2:
        second_candidates = [h for h in agg_hands[1:] if h != hour]
        if second_candidates:
            second = max(second_candidates, key=lambda h: h['length'])

    print(f"Minute hand angle: {minute['angle']:.1f}")
    print(f"Hour hand angle: {hour['angle']:.1f}")
    if second:
        print(f"Second hand angle: {second['angle']:.1f}")

    def angle_to_time(angle, divisions):
        return (angle / 360) * divisions

    minute_value = angle_to_time(minute['angle'], 60)
    hour_value = angle_to_time(hour['angle'], 12)
    print(f"Estimated time: {int(hour_value)%12:02d}:{int(minute_value):02d}")
else:
    print("Not enough hands detected after merging.")

# Show result
cv2.imshow('Clock Reader', img)
cv2.waitKey(0)
cv2.destroyAllWindows()