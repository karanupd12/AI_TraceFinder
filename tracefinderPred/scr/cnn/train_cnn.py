import numpy as np
from sklearn.model_selection import train_test_split
from build_cnn import build_dual_branch

# ----------------------
# Load data
# ----------------------
X_img = np.load("processed_data/new_processed/official_images.npy")
X_noise = np.load("processed_data/new_processed/flatfield_noise.npy")
y = np.load("processed_data/new_processed/labels.npy")

print("Before processing:")
print("X_img shape:", X_img.shape)
print("X_noise shape:", X_noise.shape)
print("y shape:", y.shape)

# ----------------------
# Convert flatfield noise to grayscale (1 channel)
# ----------------------
if X_noise.shape[-1] == 3:
    X_noise = np.mean(X_noise, axis=-1, keepdims=True)  # shape -> (N, H, W, 1)

print("After grayscale conversion of X_noise:")
print("X_noise shape:", X_noise.shape)

# ----------------------
# Upsample X_noise to match X_img length
# ----------------------
if len(X_noise) < len(X_img):
    reps = len(X_img) // len(X_noise) + 1
    X_noise = np.tile(X_noise, (reps, 1, 1, 1))[:len(X_img)]

print("After upsampling X_noise:")
print("X_img shape:", X_img.shape)
print("X_noise shape:", X_noise.shape)
print("y shape:", y.shape)

# ----------------------
# Normalize inputs to [0,1]
# ----------------------
X_img = X_img.astype('float32') / 255.0
X_noise = X_noise.astype('float32') / 255.0

# ----------------------
# Train/validation split
# ----------------------
X_img_train, X_img_val, X_noise_train, X_noise_val, y_train, y_val = train_test_split(
    X_img, X_noise, y, test_size=0.2, random_state=42
)

# ----------------------
# Build dual-branch CNN
# ----------------------
model = build_dual_branch(num_classes=len(np.unique(y)))

# ----------------------
# Train model
# ----------------------
history = model.fit(
    {"official_input": X_img_train, "noise_input": X_noise_train},
    y_train,
    validation_data=(
        {"official_input": X_img_val, "noise_input": X_noise_val}, y_val
    ),
    epochs=30,
    batch_size=32
)

# ----------------------
# Save model
# ----------------------
model.save("models/dual_branch_cnn.h5")
print("Model saved!")
