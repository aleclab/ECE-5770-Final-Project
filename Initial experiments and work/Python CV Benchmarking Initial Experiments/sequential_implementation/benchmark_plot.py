import matplotlib.pyplot as plt
import csv
import os
import argparse

def plot_average_by_resolution(csv_path):
    resolution_times = []
    filter_name = "Unknown"

    capture = False
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not capture and len(row) >= 2 and row[1].lower() in {"sobel", "gaussian", "sharpen"}:
                filter_name = row[1].capitalize()
            if capture:
                if len(row) >= 2:
                    try:
                        resolution_times.append((row[0], float(row[1])))
                    except ValueError:
                        continue
            if "Resolution" in row:
                capture = True

    if not resolution_times:
        print("No average benchmark data found in file.")
        return

    # Sort by average time (ascending)
    resolution_times.sort(key=lambda x: x[1])
    resolutions, avg_times = zip(*resolution_times)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(resolutions, avg_times, color='red')
    plt.xlabel("Average Time (ms)")
    plt.ylabel("Image Resolution")
    plt.title(f"{filter_name} Filter: Average Processing Time by Image Resolution (Sorted)")
    plt.tight_layout()

    for bar, time in zip(bars, avg_times):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{time:.2f} ms", va='center')

    out_path = csv_path.replace(".csv", "_plot.png")
    plt.savefig(out_path)
    print(f"Saved sorted graph to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to benchmark CSV file")
    args = parser.parse_args()
    plot_average_by_resolution(args.csv)