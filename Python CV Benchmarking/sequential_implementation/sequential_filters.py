import cv2
import numpy as np
import os
import time
import csv
from collections import defaultdict

def sobel_filter(img):
    channels = cv2.split(img)
    result_channels = []
    for ch in channels:
        sobel_x = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(sobel_x, sobel_y)
        grad = cv2.convertScaleAbs(grad)
        result_channels.append(grad)
    return cv2.merge(result_channels)

def gaussian_filter(img):
    return cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)

def sharpen_filter(img):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

def side_by_side(original, filtered):
    return np.hstack((original, filtered))

def apply_filter_to_folder(folder, filter_name):
    filter_funcs = {
        "sobel": sobel_filter,
        "gaussian": gaussian_filter,
        "sharpen": sharpen_filter
    }

    if filter_name not in filter_funcs:
        print(f"Invalid filter name '{filter_name}'. Choose from: {list(filter_funcs.keys())}")
        return

    output_folder = f"{filter_name}_sequential_ouput"
    os.makedirs(output_folder, exist_ok=True)
    benchmark_file = os.path.join(output_folder, f"{filter_name}_benchmark.csv")

    dimension_times = defaultdict(list)

    with open(benchmark_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Filter", "Width", "Height", "Time_ms"])

        for filename in os.listdir(folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(folder, filename)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Failed to read image: {path}")
                    continue

                h, w = img.shape[:2]
                start = time.time()
                output = filter_funcs[filter_name](img)
                end = time.time()
                elapsed_ms = 1000 * (end - start)

                print(f"Processed {filename} ({w}x{h}) in {elapsed_ms:.2f} ms")
                dimension_times[(w, h)].append(elapsed_ms)
                writer.writerow([filename, filter_name, w, h, f"{elapsed_ms:.2f}"])

                base_name = os.path.splitext(filename)[0]
                filtered_name = f"{base_name}_{filter_name}.png"
                comparison_name = f"{base_name}_{filter_name}_side_by_side.png"

                cv2.imwrite(os.path.join(output_folder, filtered_name), output)
                side_by_side_img = side_by_side(img, output)
                cv2.imwrite(os.path.join(output_folder, comparison_name), side_by_side_img)

        writer.writerow([])
        writer.writerow(["Resolution", "Average_Time_ms"])
        for (w, h), times in sorted(dimension_times.items()):
            avg_time = sum(times) / len(times)
            writer.writerow([f"{w}x{h}", f"{avg_time:.2f}"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing images")
    parser.add_argument("filter", help="Filter to apply: sobel | gaussian | sharpen")
    args = parser.parse_args()
    apply_filter_to_folder(args.folder, args.filter)