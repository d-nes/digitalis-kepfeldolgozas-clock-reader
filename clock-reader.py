import cv2
import numpy as np

def add_salt_pepper_noise(image, amount=0.01, salt_vs_pepper=0.5):
    """Add salt and pepper noise to image."""
    noisy = image.copy()
    num_pixels = image.size * amount
    # Salt noise
    num_salt = int(num_pixels * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # Pepper noise
    num_pepper = int(num_pixels * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_gaussian_noise(image, mean=0, sigma=10):
    """Add Gaussian noise to image."""
    gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy = image.astype('float32') + gauss
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy

# --- Noise selection UI ---
print("Noise options:")
print("  0 - No noise")
print("  1 - Add salt & pepper noise")
print("  2 - Add Gaussian noise")
noise_choice = input("Select noise type (0/1/2): ").strip()

img = cv2.imread('test-images/faliora.jpg')
if noise_choice == "1":
    gray_for_noise = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noisy = add_salt_pepper_noise(gray_for_noise, amount=0.02)
    img = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    print("Salt & pepper noise added.")
elif noise_choice == "2":
    gray_for_noise = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noisy = add_gaussian_noise(gray_for_noise, sigma=20)
    img = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    print("Gaussian noise added.")
else:
    print("No noise added.")

# Load and preprocess image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Detect clock face (circle)
circles = cv2.HoughCircles(
    blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
    param1=50, param2=30, minRadius=100, maxRadius=0
)
if circles is not None:
    circles = np.uint16(np.around(circles))
    i = circles[0, 0]  # Use the largest/first circle
    center = (i[0], i[1])
    radius = i[2]
    print(f"Detected clock face center: {center}, radius: {radius}")
    cv2.circle(img, center, radius, (0, 255, 0), 2)
else:
    print("No clock face detected.")
    exit()

# Edge detection
edges = cv2.Canny(blur, 30, 100, apertureSize=3)

# Detect lines (potential clock hands)
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
        # Keep lines close to the center (likely hands)
        if dist1 < radius * 0.3 or dist2 < radius * 0.3:
            all_lines.append((x1, y1, x2, y2, length))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print("  -> [kept]")
        else:
            print("  -> [discarded]")
else:
    print("No lines detected.")

print(f"Total kept hand candidates: {len(all_lines)}")

# Calculate angle and length for each hand candidate
hand_infos = []
for (x1, y1, x2, y2, length) in all_lines:
    d1 = np.hypot(x1 - center[0], y1 - center[1])
    d2 = np.hypot(x2 - center[0], y2 - center[1])
    tip = (x1, y1) if d1 > d2 else (x2, y2)
    dx = tip[0] - center[0]
    dy = center[1] - tip[1]
    angle = np.degrees(np.arctan2(dx, dy)) % 360  # 0 is up, increases clockwise
    hand_infos.append({'angle': angle, 'length': length, 'tip': tip})
    print(f"Hand candidate: angle={angle:.1f}, length={length:.1f}")

def cluster_lines_by_angle(hand_infos, angle_thresh=10):
    """Group hand candidates by similar angle."""
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

# Aggregate clusters: average angle and length
agg_hands = []
for idx, cluster in enumerate(clusters):
    avg_angle = np.mean([h['angle'] for h in cluster])
    avg_length = np.mean([h['length'] for h in cluster])
    agg_hands.append({'angle': avg_angle, 'length': avg_length, 'count': len(cluster)})
    print(f"Cluster {idx}: avg_angle={avg_angle:.1f}, avg_length={avg_length:.1f}, count={len(cluster)}")

# Sort by length (ascending: hour < minute < second)
agg_hands = sorted(agg_hands, key=lambda h: h['length'])
# If 4 clusters, drop the shortest (likely noise)
if len(agg_hands) == 4:
    min_length = min(h['length'] for h in agg_hands)
    agg_hands = [h for h in agg_hands if h['length'] > min_length + 1e-3]
    print("Dropped shortest cluster.")

# Compute angle deviation for each cluster
for h in agg_hands:
    h['angle_std'] = 0
for idx, cluster in enumerate(clusters):
    if len(cluster) > 1:
        std = np.std([h['angle'] for h in cluster])
        avg_angle = np.mean([h['angle'] for h in cluster])
        for agg in agg_hands:
            if abs(agg['angle'] - avg_angle) < 1e-3:
                agg['angle_std'] = std

# Assign hands based on angle deviation and length
agg_hands = sorted(agg_hands, key=lambda h: h['angle_std'])
second = agg_hands[0]
rest = [h for h in agg_hands if h != second]
minute = max(rest, key=lambda h: h['length'])
hour = min(rest, key=lambda h: h['length'])

print(f"Minute hand angle: {minute['angle']:.1f}")
print(f"Hour hand angle: {hour['angle']:.1f}")
print(f"Second hand angle: {second['angle']:.1f}")

def angle_to_time(angle, divisions):
    """Convert hand angle to time value (minutes or hours)."""
    return (angle / 360) * divisions

minute_value = angle_to_time(minute['angle'], 60)
hour_value = angle_to_time(hour['angle'], 12)
print(f"Estimated time: {int(hour_value)%12:02d}:{int(minute_value):02d}")

# Show result
cv2.imshow('Clock Reader', img)
cv2.waitKey(0)
cv2.destroyAllWindows()