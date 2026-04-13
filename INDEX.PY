"""
Name: Lakshay
Roll No: 2301010306
Course: Image Processing & Computer Vision
Unit: Mini Project
Assignment Title: Noise Modeling and Image Restoration using Python
Date:
"""

import cv2
import numpy as np
import os
import math

print("📷 Image Restoration for Surveillance Camera Systems")

# Create outputs folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# -----------------------------
# Task 1: Load & Preprocess
# -----------------------------
image_path = "sample.jpg"  # change this
img = cv2.imread(image_path)

if img is None:
    print("❌ Error: Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("outputs/original.png", gray)

print("✅ Original image loaded and converted to grayscale")

# -----------------------------
# Task 2: Noise Modeling
# -----------------------------

# Gaussian Noise
def add_gaussian_noise(image):
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy

# Salt & Pepper Noise
def add_salt_pepper_noise(image):
    noisy = image.copy()
    prob = 0.02

    # Salt
    num_salt = int(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    # Pepper
    num_pepper = int(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

gaussian_noise = add_gaussian_noise(gray)
sp_noise = add_salt_pepper_noise(gray)

cv2.imwrite("outputs/gaussian_noise.png", gaussian_noise)
cv2.imwrite("outputs/salt_pepper_noise.png", sp_noise)

print("✅ Noise added (Gaussian & Salt-Pepper)")

# -----------------------------
# Task 3: Filtering
# -----------------------------

# Mean Filter
mean_gaussian = cv2.blur(gaussian_noise, (5, 5))
mean_sp = cv2.blur(sp_noise, (5, 5))

# Median Filter
median_gaussian = cv2.medianBlur(gaussian_noise, 5)
median_sp = cv2.medianBlur(sp_noise, 5)

# Gaussian Filter
gauss_gaussian = cv2.GaussianBlur(gaussian_noise, (5, 5), 0)
gauss_sp = cv2.GaussianBlur(sp_noise, (5, 5), 0)

# Save restored images
cv2.imwrite("outputs/mean_gaussian.png", mean_gaussian)
cv2.imwrite("outputs/median_gaussian.png", median_gaussian)
cv2.imwrite("outputs/gaussian_gaussian.png", gauss_gaussian)

cv2.imwrite("outputs/mean_sp.png", mean_sp)
cv2.imwrite("outputs/median_sp.png", median_sp)
cv2.imwrite("outputs/gaussian_sp.png", gauss_sp)

print("✅ Filtering applied")

# -----------------------------
# Task 4: Metrics
# -----------------------------

def mse(original, restored):
    return np.mean((original - restored) ** 2)

def psnr(original, restored):
    mse_val = mse(original, restored)
    if mse_val == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse_val))

# Evaluate Gaussian Noise
print("\n📊 Performance for Gaussian Noise:")
filters_gaussian = {
    "Mean": mean_gaussian,
    "Median": median_gaussian,
    "Gaussian": gauss_gaussian
}

for name, img_restored in filters_gaussian.items():
    print(f"{name} Filter -> MSE: {mse(gray, img_restored):.2f}, PSNR: {psnr(gray, img_restored):.2f}")

# Evaluate Salt-Pepper Noise
print("\n📊 Performance for Salt & Pepper Noise:")
filters_sp = {
    "Mean": mean_sp,
    "Median": median_sp,
    "Gaussian": gauss_sp
}

for name, img_restored in filters_sp.items():
    print(f"{name} Filter -> MSE: {mse(gray, img_restored):.2f}, PSNR: {psnr(gray, img_restored):.2f}")

# -----------------------------
# Task 5: Observations
# -----------------------------

print("\n🧠 Analysis:")
print("1. Mean filter smooths image but blurs edges.")
print("2. Median filter is best for salt & pepper noise (removes impulses).")
print("3. Gaussian filter works well for Gaussian noise.")
print("4. Median filter preserves edges better than mean filter.")
print("5. Best choice depends on noise type:")
print("   - Gaussian Noise → Gaussian Filter")
print("   - Salt & Pepper → Median Filter")
