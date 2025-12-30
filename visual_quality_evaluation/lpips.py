import torch
import lpips
import os
import numpy as np
from tqdm import tqdm

# --- 1. LPIPS Model Initialization ---

use_gpu = torch.cuda.is_available()
print(f"--- LPIPS Batch Evaluation Script (Stem Matching + Auto Flip Detection) ---")
print(f"Use GPU: {use_gpu}")

# Load LPIPS model (net='alex' is usually closer to human perception)
loss_fn = lpips.LPIPS(net='alex', spatial=False)
if use_gpu:
    loss_fn.cuda()

# --- 2. Directory Configuration ---

# Root Directory A (Reference/Original Images)
root_A = "./CelebA-HQ"
# Root Directory B (Generated/Processed Images)
root_B = "./protected_image"

# Supported image extensions
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

try:
    identity_list = sorted(os.listdir(root_A))
except FileNotFoundError:
    print(f"\nError: 'root_A' directory not found: {root_A}")
    exit()

print(f"Found {len(identity_list)} identity IDs in {root_A}.")

lpips_scores = []  # Store all calculated scores
count_pairs = 0    # Counter for matched pairs

# --- 3. Traversal Logic ---

for pid in tqdm(identity_list, desc="Processing Identities"):
    folder_A = os.path.join(root_A, pid, "set_B")
    folder_B = os.path.join(root_B, pid)

    # Check if both corresponding folders exist
    if not (os.path.exists(folder_A) and os.path.exists(folder_B)):
        continue

    try:
        # Get all files in A
        images_A = sorted(os.listdir(folder_A))
        # Get all files in B
        images_B_raw = os.listdir(folder_B)
    except Exception as e:
        print(f"\nWarning: Error reading directory, skipping {pid}. Error: {e}")
        continue

    if not images_A:
        continue

    # --- Core: Build a filename index for folder B ---
    # Structure: { "filename_stem": "full_filename" }
    # E.g., if '001.jpg' is in B, dict becomes {'001': '001.jpg'}
    map_B = {}
    for fname in images_B_raw:
        if fname.lower().endswith(IMG_EXTENSIONS):
            # os.path.splitext("a.jpg") -> ("a", ".jpg")
            stem_name = os.path.splitext(fname)[0]
            map_B[stem_name] = fname

    # --- Iterate through images in A ---
    for img_name_A in images_A:
        # Filter out non-image files
        if not img_name_A.lower().endswith(IMG_EXTENSIONS):
            continue

        # Get filename stem of A (without extension)
        stem_A = os.path.splitext(img_name_A)[0]

        # Check if a matching name exists in B index (ignoring extension)
        if stem_A not in map_B:
            # Skip if no matching file found
            continue

        # Get the actual filename in B
        img_name_B = map_B[stem_A]

        path_A = os.path.join(folder_A, img_name_A)
        path_B = os.path.join(folder_B, img_name_B)

        try:
            # --- 4. LPIPS Core Calculation ---

            # Load images
            img_A_tensor = lpips.im2tensor(lpips.load_image(path_A))
            img_B_tensor = lpips.im2tensor(lpips.load_image(path_B))

            # Create a horizontally flipped version of B
            img_B_tensor_flipped = torch.flip(img_B_tensor, dims=[-1])

            if use_gpu:
                img_A_tensor = img_A_tensor.cuda()
                img_B_tensor = img_B_tensor.cuda()
                img_B_tensor_flipped = img_B_tensor_flipped.cuda()

            # Calculate normal distance
            dist_normal = loss_fn.forward(img_A_tensor, img_B_tensor)
            # Calculate flipped distance
            dist_flipped = loss_fn.forward(img_A_tensor, img_B_tensor_flipped)

            # Take the minimum
            final_dist = torch.min(dist_normal, dist_flipped)

            lpips_scores.append(final_dist.item())
            count_pairs += 1

        except Exception as e:
            print(f"\nWarning: Error processing image pair: A={img_name_A}, B={img_name_B}")
            print(f"  Error message: {e}")

# --- 5. Print Final Results ---

if not lpips_scores:
    print("\n--- Evaluation Complete (No Results) ---")
    print("No matching image pairs found (ignoring extensions).")
else:
    mean_lpips = np.mean(lpips_scores)
    print("\n" + "=" * 40)
    print(f"--- Evaluation Complete ---")
    print(f"Total pairs compared: {count_pairs} (Stem matched, ignored extensions).")
    print(f"Average LPIPS Score: {mean_lpips:.6f}")
    print("=" * 40)