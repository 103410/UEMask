import os
import numpy as np
from PIL import Image
# Remove psnr import, keep only ssim
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# --- 1. Configuration Paths ---
root_A = "./CelebA-HQ"
root_B = "./protected_image"

# Supported image extensions
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')


# --- 2. Image Loading Function ---
def load_image_as_numpy(path, size=(256, 256)):
    """
    Load image, resize, and return uint8 NumPy array in range [0, 255].
    """
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Failed to read image {path}: {e}")
        return None


# --- 3. Loop and Calculation ---
try:
    identity_list = sorted(os.listdir(root_A))
except FileNotFoundError:
    print(f"Path not found: {root_A}")
    exit()

ssim_scores = []
count_pairs = 0

print("Starting SSIM calculation (Stem Matching + Flip Check)...")

for pid in tqdm(identity_list, desc="Processing Identities"):
    # folder_A logic (Clean original images)
    folder_A = os.path.join(root_A, pid, "set_B")
    # folder_B logic (Adversarial/Protected images)
    folder_B = os.path.join(root_B, pid)

    if not (os.path.exists(folder_A) and os.path.exists(folder_B)):
        continue

    try:
        images_A = sorted(os.listdir(folder_A))
        images_B_raw = os.listdir(folder_B)
    except Exception:
        continue

    if not images_A:
        continue

    # --- Build filename index for folder B (Ignore extensions) ---
    # Structure: { "filename_stem": "full_filename" }
    map_B = {}
    for fname in images_B_raw:
        if fname.lower().endswith(IMG_EXTENSIONS):
            stem_name = os.path.splitext(fname)[0]
            map_B[stem_name] = fname

    # --- Iterate A and match ---
    for name_A in images_A:
        # Filter non-images
        if not name_A.lower().endswith(IMG_EXTENSIONS):
            continue

        # Get filename stem of A (without extension)
        stem_A = os.path.splitext(name_A)[0]

        # Check if matching stem exists in B index
        if stem_A not in map_B:
            continue

        # Get actual filename in B
        name_B = map_B[stem_A]

        path_A = os.path.join(folder_A, name_A)
        path_B = os.path.join(folder_B, name_B)

        # Load images
        img_A_np = load_image_as_numpy(path_A)
        img_B_np_original = load_image_as_numpy(path_B)

        if img_A_np is None or img_B_np_original is None:
            continue

        try:
            # --- Create horizontally flipped version of B ---
            # np.fliplr() flips along axis 1 (left-right)
            img_B_np_flipped = np.fliplr(img_B_np_original)

            # --- Calculate SSIM ---

            # 1. Compare A vs B_original (Note: channel_axis=-1 corresponds to RGB images)
            ssim_original = ssim(img_A_np, img_B_np_original, data_range=255, channel_axis=-1)

            # 2. Compare A vs B_flipped
            ssim_flipped = ssim(img_A_np, img_B_np_flipped, data_range=255, channel_axis=-1)

            # Take the higher SSIM (closer to 1 is better)
            current_ssim = max(ssim_original, ssim_flipped)

            ssim_scores.append(current_ssim)
            count_pairs += 1

        except Exception as e:
            print(f"Error calculating SSIM ({stem_A}): {e}")

# --- 4. Report Results ---
if ssim_scores:
    mean_ssim = sum(ssim_scores) / len(ssim_scores)

    print("\n==========================================")
    print(f"   Calculation complete on {count_pairs} image pairs")
    print("   (Ignored extensions, auto-detected horizontal flips)")
    print("==========================================")
    print(f"Average SSIM: {mean_ssim:.6f}    (Closer to 1 is better)")
    print("==========================================")
else:
    print("\n==========================================")
    print("No matching image pairs found for calculation.")
    print("Please check if filename stems match between root_A and root_B.")
    print("==========================================")