# ROI Light Center Extractor

---

## Project Description

This project provides a Python script to automatically process images under various lighting conditions by extracting the region of interest (ROI) centered around the brightest point. It processes images located in specific subfolders, crops them based on intensity, and saves them with a corresponding label in a CSV file. It is particularly useful for datasets where the brightest point corresponds to a relevant object or location (e.g., center of a light beam, reflection, etc.).

## Features

- **Automatic ROI Extraction**: Detects the brightest point in an image and extracts a defined ROI around it.
- **Lighting Condition Labeling**: Associates each image with a lighting level label.
- **CSV Export**: Generates a CSV file containing image filenames and their corresponding labels.
- **Folder Structure**: Automatically traverses through categorized image folders (0%, 3%, 10%, 30%, 100%) to process files.

## Folder Structure

```
project_folder/
├── img_old/
│   ├── 0%/
│   │   └── *waypoint*.jpg
│   ├── 3%/
│   └── ...
├── light/              # Output folder (auto-created)
├── output_labels.csv   # Output CSV
├── script.py
```

## Requirements

- Python 3.6 or newer

**Python Libraries**:
- `opencv-python`
- `numpy`

You can install dependencies using:
```bash
pip install opencv-python numpy
```

## Code Explanation

### ROI Extraction Logic

The core of the script lies in detecting the brightest region within a search window around the image center:

```python
def extract_roi_center(image, roi_width, roi_height, search_window=600):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_height, img_width = gray_image.shape
    center_x = img_width // 2
    center_y = img_height // 2

    # Define search area
    x_start_search = max(0, center_x - search_window // 2)
    y_start_search = max(0, center_y - search_window // 2)
    search_area = gray_image[y_start_search:y_start_search+search_window, x_start_search:x_start_search+search_window]

    # Find brightest point within search area
    sum_x = np.sum(search_area, axis=0)
    sum_y = np.sum(search_area, axis=1)
    local_center_x = np.argmax(sum_x)
    local_center_y = np.argmax(sum_y)

    # Adjust coordinates to full image
    center_x = x_start_search + local_center_x
    center_y = y_start_search + local_center_y

    # Extract ROI around detected center
    x_start = max(0, center_x - roi_width // 2)
    y_start = max(0, center_y - roi_height // 2)
    x_end = min(img_width, x_start + roi_width)
    y_end = min(img_height, y_start + roi_height)

    return image[y_start:y_end, x_start:x_end], (center_x, center_y)
```

### Main Processing Loop

Processes all images and saves cropped results:

```python
percentages = ['0%', '3%', '10%', '30%', '100%']
labels = {'0%': 0, '3%': 1, '10%': 2, '30%': 3, '100%': 4}

with open(csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['filename', 'label'])

    for percentage in percentages:
        image_pattern = os.path.join(script_dir, 'img_old', percentage, '*waypoint*.jpg')
        image_files = sorted(glob.glob(image_pattern))

        for idx, image_path in enumerate(image_files):
            image = cv2.imread(image_path)
            if image is None:
                continue

            roi, center_coords = extract_roi_center(image, 600, 700)
            waypoint_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, waypoint_name)
            cv2.imwrite(output_path, roi)

            label = labels[percentage]
            csv_writer.writerow([waypoint_name, label])
```

## Output

- Cropped ROIs saved in `light/` folder.
- CSV file `output_labels.csv` with mappings:

```csv
filename,label
waypoint_1.jpg,0
waypoint_2.jpg,1
...
```

## Notes

- The script limits its search area around the center to improve performance and robustness against background noise.
- If an image fails to load, it is skipped with a console message.

---

