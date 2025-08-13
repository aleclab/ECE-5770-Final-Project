import os
import cv2
import numpy as np
import sys 

def apply_filters(image):
    # Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Sobel Edge Detection (convert to grayscale first)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    sobel_colored = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return blurred, sobel_colored, sharpened

def save_ppm(image, path):
    cv2.imwrite(path, image)

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            in_path = os.path.join(input_dir, fname)
            image = cv2.imread(in_path)
            if image is None:
                continue
            # Save original as .ppm
            base, _ = os.path.splitext(fname)
            ppm_in = os.path.join(output_dir, f"{base}_input.ppm")
            save_ppm(image, ppm_in)
            # Apply filters
            filtered_images = apply_filters(image)
            filter_names = ['gaussian', 'sobel', 'sharpen']
            for filt_img, filt_name in zip(filtered_images, filter_names):
                out_fname = f"output_{filt_name}_{fname}"
                out_path = os.path.join(output_dir, out_fname)
                cv2.imwrite(out_path, filt_img)
                # Save as .ppm
                ppm_out = os.path.join(output_dir, f"{base}_output_{filt_name}.ppm")
                save_ppm(filt_img, ppm_out)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <input_dir> <output_dir>")
        sys.exit(1)
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    process_images(input_directory, output_directory)