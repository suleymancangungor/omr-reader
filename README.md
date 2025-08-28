# Optical Mark Recognition (OMR) System

This project is an **Optical Mark Recognition (OMR) system** designed to read and evaluate scanned or photographed exam sheets. It's built to be robust, handling both **color** and **grayscale** forms while performing **perspective correction** and accurately detecting **answer bubbles**. The system is capable of evaluating results across multiple subjects.

---

## Features

- **OMR Evaluation**: Reads and scores exam sheets.
- **Color & Grayscale Support**: Works with various image types.
- **Perspective Correction**: Automatically straightens distorted images.
- **Bubble Detection**: Accurately finds and identifies answer bubbles.
- **Multi-Subject Evaluation**: Handles scoring for multiple subjects on one sheet.

---

## Future Improvements

- **Adaptive Bubble Size**: Automatically adjust to different bubble sizes.
- **Configurable Layouts**: Allow users to define custom exam sheet layouts.
- **Data Export**: Export results to popular formats like CSV or Excel.
- **Improved Bubble Handling**: Better detection of partially filled bubbles.
- **Performance Optimization**: Enhance processing speed for large batches.

---

## Usage

You can run the OMR system in two different ways:

### 1. Run with a single image

To process a single image, use the `--image` flag in your terminal:

`python main.py --image path/to/input.jpg`

This will process only the specified image and display the results.

### 2. Run in batch mode

To process multiple images at once, follow these steps:

1.  Place all your exam sheet images (supported formats: `.jpg`, `.jpeg`, `.png`) inside the `images/` folder.
2.  In the `main.py` file, ensure the `image_folder` variable is set to the correct path:

    ```python
    image_folder = "path/to/your/images"
    ```
3.  Run the script from your terminal or IDE:

    ```
    python main.py
    ```

This will automatically process all images in the specified folder.

---

## Requirements

The project requires **Python 3.8+** and the following libraries:

-   `opencv-python`
-   `numpy`
-   `imutils`

You can install all dependencies by running the following command:

`pip install -r requirements.txt`

---

## Author

Developed by **Süleyman Can Güngör**

-   [GitHub Profile](https://github.com/suleymancangungor)

