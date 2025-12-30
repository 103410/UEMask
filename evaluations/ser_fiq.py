import cv2
import argparse
import os
from collections import defaultdict
from FaceImageQuality.face_image_quality import SER_FIQ


def parse_args():
    """
    The --root_dir argument should point to the root directory containing
    all identity folders (e.g., 103, 104, etc.).
    """
    parser = argparse.ArgumentParser(description='Aggregate Face Image Quality Assessment Across Identities')
    parser.add_argument('--root_dir',
                        default='./test_infer/',
                        help='Path to the root directory containing all identity folders')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    """
    Main function to calculate the average face image quality score for each
    text prompt across all identities.
    """
    args = parse_args()
    MAX_IDENTITIES = 10

    # Check if root directory exists
    if not os.path.isdir(args.root_dir):
        print(f"Error: Root directory does not exist -> {args.root_dir}")
        return

    print(f"Start processing root directory: {args.root_dir}")
    print("-" * 40)

    # Initialize the model
    ser_fiq = SER_FIQ(gpu=args.gpu)

    # Use defaultdict to store all scores corresponding to each prompt
    # Structure: {'prompt_name_1': [score1, score2, ...], 'prompt_name_2': [scoreA, scoreB, ...]}
    prompt_scores_aggregator = defaultdict(list)

    # 1. Iterate through all identity folders
    all_identity_folders = sorted(
        [d for d in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, d))])
    identity_folders = all_identity_folders[:MAX_IDENTITIES]

    for identity_name in identity_folders:
        # identity_path = os.path.join(args.root_dir, identity_name, 'checkpoint-1000','dreambooth')
        identity_path = os.path.join(args.root_dir, identity_name)
        print(f"Processing identity: {identity_name}...")

        # 2. Iterate through all text prompt folders under this identity
        for prompt_name in os.listdir(identity_path):
            prompt_path = os.path.join(identity_path, prompt_name)
            if not os.path.isdir(prompt_path):
                continue

            # 3. Iterate through all images in the prompt folder
            for img_name in os.listdir(prompt_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(prompt_path, img_name)
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"  - Warning: Unable to read image {img_path}")
                            continue

                        aligned_img = ser_fiq.apply_mtcnn(img)
                        if aligned_img is not None:
                            score = ser_fiq.get_score(aligned_img, T=100)
                            # Add the score to the list corresponding to the prompt
                            prompt_scores_aggregator[prompt_name].append(score)

                    except Exception as e:
                        print(f"  - Error processing image {img_path}: {e}")

    print("-" * 40)
    print("Processing of all identities completed. Calculating final average scores...")
    print("-" * 40)

    # 4. Calculate and print the final average score for each prompt
    if not prompt_scores_aggregator:
        print("No processable images found.")
        return

    for prompt_name, scores_list in sorted(prompt_scores_aggregator.items()):
        if scores_list:
            average_score = sum(scores_list) / len(scores_list)
            print(f"Text Prompt: {prompt_name}")
            print(f"  - Total image count: {len(scores_list)}")
            print(f"  - Average FIQ score across all identities: {average_score:.4f}\n")
        else:
            # Theoretically, this won't happen since we only append when there is a score
            print(f"Text Prompt: {prompt_name}\n  - No valid scores collected.\n")


if __name__ == '__main__':
    main()