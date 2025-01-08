from argparse import ArgumentParser
from pathlib import Path
from simulated_annealing import find_best_gripper_position

import pandas as pd

from rich.progress import track


def main():
    """
    Solution for the ProKI Hackathon 2024.
    Pipeline:
    1. Predict a grayscale mask of the part image with a fine-tuned DeepLabV3Plus model
    2. Post-process the output with thresholding and erosion to get a binary mask
    3. Compute the best gripper position on the binary mask using simulated annealing
    """

    parser = ArgumentParser()
    parser.add_argument("input", help="input csv file")
    parser.add_argument("output", help="output csv file")
    args = parser.parse_args()

    # read the input csv file
    input_df = pd.read_csv(args.input)

    # compute the solution for each row
    results = []
    for _, row in track(input_df.iterrows(), description="Computing the solutions for each row", total=len(input_df), ):
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
        x, y, angle = find_best_gripper_position(part_image_path, gripper_image_path)
        results.append([str(part_image_path), str(gripper_image_path), x, y, angle])

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle"])
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
