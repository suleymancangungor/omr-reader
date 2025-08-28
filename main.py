import argparse
import numpy as np
import imutils
import cv2
import os
import math


# ----------------------------------------------------------------------
# Main entry point for running the OMR system.
#
# Usage:
#   - To run the code on a single image:
#       python main.py --image path/to/input.jpg
#
#   - To process multiple images at once:
#       1. Set 'image_folder' to the folder containing the images.
#       2. Ensure supported formats are listed in 'supported_formats'.
#       3. Uncomment the loop at the bottom of the script.
#       4. Run:
#           python main.py
#
# Notes:
#   - The script performs perspective correction, detects alignment bars,
#     finds answer columns, checks answers, extracts student ID and booklet type.
#   - Results are printed to the console and can optionally be written
#     to a text file (see commented-out section with 'output_txt').
#   - After each run, results (correct/wrong counts) are reset to avoid
#     carry-over when processing multiple forms.
# ----------------------------------------------------------------------


def perspective_transform(image, crop_ratio=0.1, min_r = 15, max_r=25, circ_thr=0.7):
    """
        Apply a perspective transformation to the exam sheet image so that
        the form is deskewed and aligned to a rectangular grid.

        Process:
        - Converts the image to grayscale and crops the top region 
            (defined by `crop_ratio`) to focus on the bubble alignment markers.
        - Applies Gaussian blur and adaptive thresholding to detect contours.
        - Uses contour analysis and circularity checks to find circular markers 
            (bubbles) that serve as corner reference points.
        - Identifies the four corner points (top-left, top-right, bottom-left, bottom-right).
        - Computes a perspective transform matrix (cv2.getPerspectiveTransform) 
            to warp the original image into a straightened, rectangular view.
        - Applies an offset to ensure the warped image has positive coordinates 
            and fills outside regions with white.

        Parameters:
        image (numpy.ndarray): Input exam sheet image (BGR).
        crop_ratio (float): Fraction of the top of the image to ignore 
                            when searching for reference bubbles 
                            (e.g., to avoid headers).
        min_r, max_r (int): Minimum and maximum radius of valid corner bubbles.
        circ_thr (float): Minimum circularity threshold to accept a contour 
                            as a valid circle.

        Returns:
        warped (numpy.ndarray): The perspective-corrected (deskewed) version 
                                of the input image.
        corner_pts (list of tuples): The four detected corner points 
                                    [top-left, top-right, bottom-right, bottom-left].

        Notes:
        - If no valid corner circles are found, the function exits.
        - This step is critical for OMR since it ensures that answer bubbles 
            align consistently across different scanned or photographed sheets.
        - The use of fixed values for `min_r` and `max_r` can be limiting, 
            since bubble sizes vary with image resolution and scaling. 
            Ideally, these should be adaptive (e.g., relative to image height/width 
            or estimated bubble size) for more robust handling of diverse inputs.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    gray = gray[int(h*crop_ratio):, :]
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for c in cnts:
        (x, y), r = cv2.minEnclosingCircle(c)
        x,y,r = int(x), int(y), int(r)
        if r < min_r or r > max_r:
            continue

        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * math.pi * area / (peri * peri)
        if circularity < circ_thr:
            continue

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        roi_vals = gray[mask == 255]
        bubbles.append((x, y+int(h*crop_ratio), r))

    if len(bubbles) == 0:
        print("No circles found. Exiting.")
        exit()

    coords = np.array([(x,y) for (x,y,_) in bubbles])
    sum_xy = coords[:, 0] + coords[:, 1]
    diff_xy = coords[:, 0] - coords[:, 1]

    tl = coords[np.argmin(sum_xy)]
    tr = coords[np.argmax(diff_xy)]
    bl = coords[np.argmin(diff_xy)]
    br = np.array([tr[0]+bl[0]-tl[0], tr[1]+bl[1]-tl[1]])
    # br = coords[np.argmax(sum_xy)]
    corner_pts = [tl,tr,br,bl]

    pts_src = np.array([tl,tr,br,bl], dtype='float32')
    width = int((np.linalg.norm(tr-tl) + np.linalg.norm(br-bl)) / 2)
    height = int((np.linalg.norm(bl-tl) + np.linalg.norm(br-tr)) / 2)
    pts_dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    h, w = image.shape[:2]
    corners = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype='float32').reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, M)
    
    [x_min, y_min] = np.floor(np.min(transformed_corners, axis=0)[0])
    [x_max, y_max] = np.ceil(np.max(transformed_corners, axis=0)[0])
    new_w = int(x_max - x_min)
    new_h = int(y_max - y_min)

    offset = np.array([[1, 0, -x_min],[0, 1, -y_min],[0, 0, 1]])
    M_offset = offset @ M
    M_offset = M_offset[:3]
    
    warped = cv2.warpPerspective(image, M_offset, (new_w, new_h), 
                            flags=cv2.INTER_LANCZOS4, 
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255,255,255))
                            
    return warped, corner_pts 


def remove_pure_black_areas(image, black_threshold):
    """
        Replace near-black pixel regions in the image with white to reduce noise.

        Args:
            image (numpy.ndarray):
                Input grayscale image.
            black_threshold (int):
                Pixel intensity threshold (0-255). 
                Pixels with values less than or equal to this threshold 
                are considered "pure black" and will be set to white.

        Returns:
            cleaned_image (numpy.ndarray):
                Copy of the input image where pure black areas are replaced by white (255).

        Notes:
            - Useful for cleaning up scanner artifacts or solid black areas 
            that might otherwise be mistaken for alignment marks or bubbles.
            - Typical values of `black_threshold` are low (e.g., 5-15).
    """
    cleaned_image = image.copy()
    black_mask = image <= black_threshold
    cleaned_image[black_mask] = 255
    return cleaned_image


def find_alignment_bar_y_coords(gray, bound):
    """
        Detect the horizontal alignment marks (small black bars on the left side of the form)
        and return their y-coordinates for row alignment.

        Args:
            gray (numpy.ndarray):
                Grayscale version of the exam sheet image.
            bound (int):
                Right boundary (in pixels) of the region to search for the alignment bar.
                Typically a slice from the left edge up to this bound.

        Returns:
            y_coords (list of int):
                List of y-coordinates representing the centers of detected alignment marks.

        Notes:
            - The function first crops the left margin of the form (up to `bound`).
            - It applies Gaussian blur and Otsu thresholding to highlight dark marks.
            - Morphological closing is used to strengthen rectangular shapes.
            - Contours are extracted, and bounding boxes are filtered by:
                * aspect ratio (wide and short rectangles),
                * extent (area ratio),
                * relative height constraints based on image size.
            - Valid boxes are considered alignment marks, and their vertical centers are saved.
            - `group_bubbles` is applied to merge very close detections into single coordinates.
            - These y-coordinates are later used to align rows of bubbles consistently.
    """
    alignment_bar = gray[:, 0:bound]
    alignment_bar = remove_pure_black_areas(alignment_bar, black_threshold=10)
    blur = cv2.GaussianBlur(alignment_bar, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    alignment_boxes = []
    H, W = gray.shape[:2]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        rect_area = w * h
        extent = area / float(rect_area + 1e-5)
        aspect_ratio = w/float(h)

        if area > H * W * 0.001:
            continue

        if 1.5 < aspect_ratio < 10:
            if 0.7 < extent < 1.0:
                if H*0.002 < h < H*0.02:
                    alignment_boxes.append((x,y,w,h))
                    cv2.rectangle(alignment_bar, (x,y), (x+w, y+h), (0, 255, 0), 2)

    y_coords = sorted([y + h//2 for (_,y,_,h) in alignment_boxes])
    y_coords = group_bubbles(y_coords, threshold=int(H*0.005))

    if not y_coords:
        print("Alignment bar couldn't found.")

    return y_coords


def dedup_bubbles(bubbles):
    """
        Merge duplicate bubble detections by grouping overlapping circles.

        Args:
            bubbles (list of tuples): 
                List of detected bubbles in the form (x, y, r), 
                where x and y are the center coordinates and r is the radius.

        Returns:
            out (list of tuples): 
                Deduplicated list of bubbles, where overlapping detections 
                are merged into a single bubble with averaged center (x, y) 
                and radius r.

        Notes:
            - Sometimes the same bubble may be detected multiple times 
            (e.g., once by contours and once by HoughCircles).
            - This function checks the distance between circles, and if 
            they overlap (distance < r1 + r2), they are considered the same.
            - The final bubble is placed at the average position with an 
            averaged radius to reduce noise.
    """
    out = []
    used = [False] * len(bubbles)
    for i, (x,y,r) in enumerate(bubbles):
        if used[i]:
            continue
        group = [(x,y,r)]
        used[i] = True
        for j, (x2,y2,r2) in enumerate(bubbles):
            if used[j]:
                continue
            dist = math.hypot(x-x2, y-y2)
            if dist < (r + r2):
                group.append((x2,y2,r2))
                used[j] = True

        xs,ys,rs = zip(*group)
        avg_x = int(np.mean(xs))
        avg_y = int(np.mean(ys))
        avg_r = int(np.mean(rs))
        out.append((avg_x, avg_y, avg_r))

    return out


def circular_roi(gray, x, y, r, return_vals=False):
    """
        Extracts a circular region of interest (ROI) from a grayscale image.

        Args:
            gray (ndarray): Input grayscale image.
            x (int): X-coordinate of the circle center.
            y (int): Y-coordinate of the circle center.
            r (int): Radius of the circle.
            return_vals (bool, optional): 
                If True, also returns the pixel intensity values inside the circle. 
                Default is False.

        Returns:
            circ (ndarray): The circular region extracted from the image 
                            (pixels outside the circle are set to white).
            roi_vals (ndarray, optional): 
                Array of pixel intensities inside the circle (only if return_vals=True).

        Notes:
            - This function masks out a circular region from the input image, 
            preserving only the pixels inside the given radius.
            - The outside area of the circle is filled with white (255).
            - Useful for analyzing bubble fill ratios in OMR systems by isolating
            the marked/unmarked area of each bubble.
    """
    h, w = gray.shape[:2]
    y1, y2 = max(0, y-r), min(h, y+r)
    x1, x2 = max(0, x-r), min(w, x+r)

    sub = gray[y1:y2, x1:x2].copy()
    mask = np.zeros_like(sub, dtype=np.uint8)

    cy, cx = y - y1, x - x1
    cv2.circle(mask, (cx, cy), r, 255, -1)
    circ = cv2.bitwise_and(sub, sub, mask=mask)
    outside_mask = cv2.bitwise_not(mask)
    circ[outside_mask == 255] = 255

    if return_vals:
        roi_vals = circ[mask == 255]
        return circ, roi_vals
    return circ


def find_bubbles_on_row(image, y_center, x_bound=0, row_height=40, min_r=17, max_r=30, black_thr=0.5):
    """
        Detects all bubbles along a single row (both filled and unfilled) 
        by analyzing contours and circularity.

        Process:
        - Crops a horizontal strip around the given row (y_center ± row_height/2).
        - Converts to grayscale, applies Gaussian blur, and thresholds the image 
            to highlight bubble regions.
        - Finds contours and approximates each with a minimum enclosing circle.
        - Filters candidates based on circularity and radius range (min_r, max_r).
        - Extracts a circular ROI for each candidate and computes the black pixel ratio 
            using Otsu thresholding. If the ratio exceeds `black_thr`, the bubble is 
            considered filled/valid.
        - Additionally, applies HoughCircles as a fallback to ensure robust detection.
        - Deduplicates and sorts bubbles by their x-coordinate.

        Parameters:
        image      : Input exam sheet image (BGR).
        y_center   : Center y-coordinate of the row to scan.
        x_bound    : Left boundary cutoff (used to ignore unwanted dark areas or 
                    alignment marks on the left side).
        row_height : Height of the cropped band for bubble detection. 
                    (Note: a fixed value may cause issues across different 
                    resolutions; adaptive adjustment is recommended.)
        min_r/max_r: Minimum and maximum bubble radius for filtering. 
                    (These are fixed values, which may need scaling for different 
                    image sizes or resolutions.)
        black_thr  : Threshold for the black pixel ratio inside the bubble 
                    (values above this indicate a marked/filled bubble).

        Returns:
        bubbles : A sorted list of detected bubbles as tuples (x, y, r).
                    Each bubble is represented by its center (x,y) and radius r.
    """
    bubbles = []
    h, w = image.shape[:2]
    y1 = max(y_center - row_height//2, 0)
    y2 = min(y_center + row_height//2, h)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row_crop = image[y1:y2, x_bound:]   
    gray_crop = cv2.cvtColor(row_crop, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray_crop, (3,3), 0)
    # _, thresh_otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_fixed = cv2.threshold(gray_blur, bubble_threshold, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(thresh_fixed, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_crop = row_crop.copy()

    for c in cnts:
        (x,y), r = cv2.minEnclosingCircle(c)
        circle_area = np.pi * (r**2)
        area = cv2.contourArea(c)
        circularity = area / circle_area 

        if 0.5 < circularity < 1.2 and min_r < r < max_r:
            roi, roi_vals = circular_roi(gray, int(x+x_bound), int(y+y1), int(r), return_vals=True)
            otsu_thresh, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = np.zeros_like(roi, dtype=np.uint8)
            cv2.circle(mask, (roi.shape[1]//2, roi.shape[0]//2), int(r*0.8), 255, -1)
            roi_gray = roi[mask == 255]
            black_ratio = np.sum(roi_gray < otsu_thresh) / roi_gray.size
            if black_ratio > black_thr:
                bubbles.append((int(x+x_bound), int(y+y1), int(r)))
                # cv2.circle(debug_crop, (int(x), int(y)), int(r), (0, 255, 0), 2)

    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=80, param2=40, minRadius=min_r, maxRadius=max_r)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x,y,r) in circles:
            bubbles.append((int(x+x_bound), int(y+y1), int(r)))
            # cv2.circle(debug_crop, (int(x), int(y)), int(r), (0, 255, 0), 2)

    # cv2.imshow("row cropped", debug_crop)
    # cv2.waitKey(0)

    bubbles = dedup_bubbles(bubbles)
    bubbles = sorted(bubbles, key=lambda b: b[0])
    # print("FBOR: ", len(bubbles))

    return bubbles


def find_columns_xs_grouped(image, y_coords, start_index, row_count, group_size, x_bound, start=None, end=None):
    """
        Detects bubbles across multiple rows and groups their x-coordinates to 
        reliably determine the column positions.

        Purpose:
        - Avoids relying on a single row, since a bubble may be faint, crossed out,
            or noisy. By scanning multiple rows, the column x-coordinates can be 
            identified more robustly.
        - Uses group_bubbles() to merge nearby x-coordinates so that each column 
            is represented by one averaged x-position.
        - Optional `start` and `end` parameters allow excluding areas on the left 
            or right (e.g., student ID or booklet type regions) so that only answer 
            columns are detected.

        Parameters:
        image       : The exam sheet image to process.
        y_coords    : List of y-coordinates (row centers) for the questions.
        start_index : Index of the first row to start scanning from.
        row_count   : Number of rows to scan for bubble detection.
        group_size  : Expected number of bubbles per row/column group.
        x_bound     : Left boundary to ignore alignment marks or side codes.
        start/end   : Limit the range of grouped x-coordinates considered.
                        Useful if the form has extra coding areas outside 
                        the main answer columns.

        Returns:
        columns : List of grouped x-coordinate lists, one list per column.
        avg_r   : Average radius of detected bubbles.
    """
    xs = []
    radii = []
    for i in range(start_index, start_index+row_count):
        bubbles = find_bubbles_on_row(image, y_coords[i], x_bound)
        for (x,y,r) in bubbles:
            xs.append(x)
            radii.append(r)
            # cv2.circle(image, (x,y), r, (0, 0, 255), 2)
    
    if not xs:
        print(f"No bubbles found in rows {start_index} to {start_index+row_count-1}.")
        return [], 0
    avg_r = int(np.mean(radii))
    r_threshold = int(avg_r * 1.2)

    grouped = group_bubbles(xs, r_threshold)
    if end is None:
        end = len(grouped)
    if start is None:
        start = 0
    grouped_sorted = sorted(grouped)[start:end]
    columns = [grouped_sorted[i:i+group_size] for i in range(0, len(grouped_sorted), group_size)]
    # print(f"Columns found: {columns}")
    return columns, avg_r    


def group_bubbles(coords, threshold):
    """
        Takes a list of coordinates (e.g., x or y),
        groups those that are within the given threshold distance,
        and returns the average value of each group.
        This way, multiple detections of the same bubble
        are reduced to a single representative coordinate.
    """
    if not coords:
        return []
    coords = sorted(coords)
    grouped = []
    current_group = [coords[0]]

    for c in coords[1:]:
        if abs(c-current_group[-1]) <= threshold:
            current_group.append(c)
        else:
            grouped.append(int(sum(current_group)/len(current_group)))
            current_group = [c]
    grouped.append(int(sum(current_group)/len(current_group)))
    return grouped


def crop_column_region(image, columns, avg_r, padding_factor=2.0):
    """
        Crop the image to include only the specified bubble columns, 
        with optional horizontal padding.

        Purpose:
        - Focuses on a specific set of columns (answer choices) to reduce 
            processing overhead and avoid interference from other parts of the form.
        - Useful when multiple columns of bubbles exist and checking needs 
            to be restricted to a particular subset.
        - For performance optimization, this step could be replaced with 
            direct coordinate-based indexing in other functions instead of 
            cropping the entire region.

        Parameters:
        image (numpy.ndarray): Input exam sheet image.
        columns (list of int): X-coordinates of bubble columns to be included.
        avg_r (int): Average bubble radius, used to calculate padding.
        padding_factor (float, optional): Multiplier for the radius to 
                                            determine extra margin on both sides.

        Returns:
        cropped (numpy.ndarray): Cropped image containing the specified columns.
        x_min (int): Left offset of the cropped region relative to the original image.

        Notes:
        - If no columns are provided, the original image is returned unchanged.
        - Padding ensures bubbles near the edges of a column are not cut off.
        - For more efficient processing, cropping can be skipped and handled 
            directly by using column coordinate ranges.
    """
    if not columns:
        print("No columns provided for cropping.")
        return image, 0
    padding = int(avg_r * padding_factor)
    x_min = max(min(columns)-padding, 0)
    x_max = min(max(columns)+padding, image.shape[1])
    cropped = image[:, x_min:x_max]
    return cropped, x_min


def check_answers(image, gray, columns, y_coords, avg_r, fill_thr=0.3):
    """
        Check the answers by detecting marked bubbles and comparing them 
        with the provided answer keys.

        Process:
        - Iterates through each subject/column group.
        - Crops the corresponding column region (using crop_column_region).
        - For each question row, calls find_bubbles_on_row to detect bubbles.
        - Calculates the fill ratio for each bubble using circular_roi.
        - Decides whether a bubble is marked or not based on fill_thr 
            (with "strong" marks given higher confidence).
        - Compares the detected marked bubble(s) with the answer key:
            * If a single bubble matches the correct index → mark as correct.
            * If a wrong bubble is chosen → mark as wrong.
            * If multiple bubbles are filled strongly → mark as invalid/ambiguous.

        Parameters:
        image (numpy.ndarray): Original exam sheet image (will be annotated with circles).
        gray (numpy.ndarray): Grayscale version of the exam sheet.
        columns (list): Grouped x-coordinates for bubble columns.
        y_coords (list): Y-coordinates for question rows.
        avg_r (int): Average bubble radius.
        fill_thr (float): Threshold for deciding whether a bubble is marked.

        Returns:
        image (numpy.ndarray): The input image annotated with colored circles:
                                green = correct, red = wrong, yellow = multiple marks.
        correct (list): List of (subject_index, question_number) tuples for correct answers.
        wrong (list): List of (subject_index, question_number) tuples for wrong answers.

        Notes:
        - This implementation currently crops column regions for each subject 
            and scans them row by row.
        - Cropping introduces extra overhead. In principle, this function can be 
            optimized to operate **directly on bubble coordinates** (x, y, r) without 
            creating cropped regions.
        - Currently, bubbles are detected twice: first by find_bubbles_on_row, 
            and then each candidate is checked again with circular_roi to calculate 
            fill ratio. These steps could be merged to reduce redundant processing 
            and make the pipeline faster.
        - The use of both 'marked' and 'strong' lists allows distinguishing 
            between faint and clearly filled bubbles.
    """
    correct = []
    wrong = []
    for index, column in enumerate(columns):
        question_count = SUBJECTS[index]["question_count"]
        answer_key = SUBJECTS[index]["answer_key"]
        cropped_column, offset = crop_column_region(image, column, avg_r)
        for i in range(question_count):
            row_index = first_question_index + i
            bubbles = find_bubbles_on_row(cropped_column, y_coords[row_index], 0)
            bubbles = [(int(x+offset), int(y), int(r)) for (x,y,r) in bubbles]
            # print("CA: ", len(bubbles))
            correct_answer_index = row_index - first_question_index

            if not bubbles:
                print(f"No bubbles found for question {i+1} in subject {SUBJECTS[index]['name']}.")
                continue

            # print(f"Checking question {i+1}")
            marked = []
            strong = []

            for i, (x,y,r) in enumerate(bubbles):
                roi, roi_vals = circular_roi(gray, x, y, r, return_vals=True)
                fill_ratio = 1 - (np.mean(roi_vals) / 255.0)
                if fill_ratio > fill_thr:
                    if fill_ratio > 0.5:
                        strong.append(i)
                    marked.append(i)

            if len(marked) == 0:
                continue # No bubbles marked
            else:
                if len(marked) > 1:
                    if len(strong) > 1:
                        for j in strong:
                            (x, y, r) = bubbles[j]
                            cv2.circle(image, (x, y), r, (0, 255, 255), 2)
                        continue   
                if len(strong) > 0:
                    chosen = strong
                else:
                    chosen = marked
                if chosen[0] == answer_key[correct_answer_index]:
                    correct.append((index, correct_answer_index+1))
                    SUBJECTS[index]["correct"] += 1
                    cv2.circle(image, (bubbles[chosen[0]][0],bubbles[chosen[0]][1]), bubbles[chosen[0]][2], (0, 255, 0), 2)
                else:
                    SUBJECTS[index]["wrong"] += 1
                    wrong.append((index, correct_answer_index+1))
                    cv2.circle(image, (bubbles[chosen[0]][0],bubbles[chosen[0]][1]), bubbles[chosen[0]][2], (0, 0, 255), 2)

    return image, correct, wrong


def check_column_based_marks(image, gray, columns, y_coords, type_dict, start_row_index, row_count, mark_count, r, fill_thr=0.3):
    """
        Detect and decode marks in column-based coding areas (e.g., student ID digits, 
        booklet type, or letter-based codes).

        Process:
        - Iterates through given columns and scans row_count rows starting at start_row_index.
        - Extracts a circular ROI at each (x, y) bubble center using circular_roi.
        - Computes fill ratio to decide whether the bubble is marked.
        - If a bubble is marked:
            * For ID_LETTER type, mapping is taken from type_dict based on row index.
            * For single-row cases, mapping is based on the column index.
            * Otherwise, the row index itself is used as the digit/character.
        - If more than one bubble is filled in the same column, a "*" symbol is placed 
            to indicate ambiguity.
        - The process continues until mark_count valid marks are detected.

        Parameters:
        image (numpy.ndarray): The exam sheet image (will be annotated with green circles).
        gray (numpy.ndarray): Grayscale version of the image.
        columns (list): List of x-coordinates for the columns to be checked.
        y_coords (list): Y-coordinates of question/row centers.
        type_dict (dict): Mapping dictionary for decoding marks (e.g., {0:'A',1:'B'} or digit map).
        start_row_index (int): Starting index in y_coords for scanning.
        row_count (int): Number of rows to check in each column.
        mark_count (int): Expected number of marks (e.g., total digits in ID).
        r (int): Bubble radius.
        fill_thr (float): Threshold for deciding if a bubble is filled.

        Returns:
        image (numpy.ndarray): Annotated image with detected marks.
        key (str): Decoded key (ID, type, or code string).

        Notes:
        - Currently, only fill ratio is used for mark detection. Combining with 
            contour analysis (e.g., black pixel ratio) could improve accuracy.
        - Multiple detections in a column are marked as "*", but more sophisticated 
            handling (e.g., warning, taking the strongest mark) could be added.
        - The logic for deciding whether to map from type_dict or row index is a bit rigid; 
            this can be refactored to make it more general and configurable.
        - At the moment, scanning stops once mark_count marks are found, which assumes 
            a fixed-length code. More flexible handling may be needed for variable-length inputs.
    """
    key = ""
    key_length = 0
    for col_index, column in enumerate(columns):
        x = column[0]
        found_digit = None
        for i in range(row_count):
            y_index = start_row_index + i
            y = y_coords[y_index]

            roi, roi_vals = circular_roi(gray, x, y, r, return_vals=True)
            if roi_vals.size == 0:
                print(f"No ROI values found for column {col_index}, row {i} at ({x}, {y}).")
                continue

            fill_ratio = 1 - (np.mean(roi_vals) / 255.0)
            if fill_ratio > fill_thr:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                if found_digit is not None:
                    found_digit = "*"
                    break
                if col_index == 0 and type_dict == ID_LETTER:
                    found_digit = type_dict.get(i, "-")
                elif row_count == 1 and type_dict != ID_LETTER:
                    found_digit = type_dict.get(col_index, "-")
                else:
                    found_digit = str(i)
                key_length += 1
                key += found_digit
        if key_length == mark_count:
            break
    return image, key


def resize_image(image):
    """
        Resize an image to fit within a fixed maximum width and height 
        while preserving its aspect ratio.

        Purpose:
        - Allows displaying large images on the screen in a smaller, 
            more manageable size (default max: 1200x800).
        - Ensures the image is scaled proportionally without distortion.

        Parameters:
        image (numpy.ndarray): The input image to resize.

        Returns:
        resized (numpy.ndarray): The resized image.

        Notes:
        - If the input image is None, the function prints an error message 
            and exits the program.
        - The resizing uses cv2.INTER_AREA interpolation, which is generally 
            best for shrinking images.
        - Current maximum dimensions are hardcoded (1200x800); these could be 
            made configurable for flexibility.
    """
    if image is None:
        print("There is no image to resize.")
        exit()
    h, w = image.shape[:2]
    maxW, maxH = 1200, 800
    scaling_factor = min(maxW/w, maxH/h)
    newW = int(w*scaling_factor)
    newH = int(h*scaling_factor)
    return cv2.resize(image, (newW, newH), interpolation=cv2.INTER_AREA)


def main(image_path, output_txt_path):
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-i", "--image", required=True, help="path to the input image")
#     args = vars(ap.parse_args())
#     image = cv2.imread(args["image"])

    image = cv2.imread(image_path)   
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image, corner_pts = perspective_transform(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # I use this to avoid detecting black areas that may appear on the edge due to the input image or perspective transform while detecting bubbles correctly
    x_bound = corner_pts[0][0]

    y_coords = find_alignment_bar_y_coords(gray, x_bound-25) # 25 is the padding to avoid the alignment bar. should be dynamic.
    answer_columns, avg_r = find_columns_xs_grouped(image, y_coords, first_question_index, answer_columns_row_count, bubble_count_per_question, x_bound, None, None)
    image, correct, wrong = check_answers(image, gray, answer_columns, y_coords, avg_r)    
    id_columns, avg_r = find_columns_xs_grouped(image, y_coords, first_id_index, id_row_count, 1, 0, None, 10)
    image, id = check_column_based_marks(image, gray, id_columns, y_coords, ID_LETTER, first_id_index, id_row_count, id_length, avg_r)
    book_type_columns, avg_r = find_columns_xs_grouped(image, y_coords, book_type_index, 1, 1, 0, None, None)
    image, book_type = check_column_based_marks(image, gray, book_type_columns, y_coords, BOOK_TYPE, book_type_index, 1, 1, avg_r)

    print("ID: ", id)
    print("Book Type: ", book_type)
    
    # with open(output_txt_path, "a", encoding="utf-8") as f:
    #     f.write("----------------------------------\n")
    #     f.write(f"{os.path.basename(image_path)}\n")
    #     f.write(f"Correct answer count of {SUBJECTS[0]["name"]}: {SUBJECTS[0]["correct"]}\n")
    #     f.write(f"Wrong answer count of {SUBJECTS[0]["name"]}: {SUBJECTS[0]["wrong"]}\n")
    #     corrects = [b for a, b in correct if a == 0]
    #     f.write(f"Corrects : {corrects}\n")
    #     wrongs = [b for a, b in wrong if a == 0]
    #     f.write(f"Wrongs : {wrongs}\n")
    #     f.write("ID: " + id + "\n")
    #     f.write("Book type: " + book_type + "\n")

    for i in range(len(SUBJECTS)):
        SUBJECTS[i]["correct"] = 0
        SUBJECTS[i]["wrong"] = 0

    image = resize_image(image)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    # exit()


if __name__ == "__main__":
    answer_columns_row_count = 40
    bubble_threshold = 150
    # We write the index of the line where the bubbles we marked the book type are located. We write the index according to the boxes in the alignment bar.
    first_question_index = 17
    bubble_count_per_question = 5
    first_id_index = 4
    id_row_count = 10
    id_length = 10
    book_type_index = 3

    SUBJECTS = [{"name":"Türkçe", "question_count": 40, "answer_key":{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
    10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1,
    20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1,
    30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1}, "correct":0, "wrong":0},
    {"name":"Sosyal", "question_count": 20, "answer_key":{0: 3, 1: 2, 2: 0, 3: 1, 4: 4, 5: 1, 6: 3, 7: 0, 8: 2, 9: 4,
    10: 0, 11: 2, 12: 1, 13: 3, 14: 4, 15: 1, 16: 0, 17: 4, 18: 3, 19: 2}, "correct":0, "wrong":0},
    {"name":"Matematik", "question_count":40, "answer_key":{0: 3, 1: 2, 2: 0, 3: 1, 4: 4, 5: 1, 6: 3, 7: 0, 8: 2, 9: 4,
    10: 0, 11: 2, 12: 1, 13: 3, 14: 4, 15: 1, 16: 0, 17: 4, 18: 3, 19: 2,
    20: 2, 21: 0, 22: 4, 23: 1, 24: 3, 25: 1, 26: 0, 27: 2, 28: 3, 29: 0,
    30: 4, 31: 1, 32: 2, 33: 3, 34: 4, 35: 0, 36: 3, 37: 1, 38: 2, 39: 4}, "correct":0, "wrong":0}]
    # I use the ID_LETTER dict because the optical papers I use can have letters at the beginning of the IDs.
    ID_LETTER = {0:"B", 1:"G", 2:"D", 3:"Y", 4:"Y", 5:"U", 6:"E", 7:"T", 8:"M"}
    BOOK_TYPE = {0:"A", 1:"B", 2:"C", 3:"D"}

    # main()

    image_folder = ""
    output_txt = "results.txt"

    supported_formats = (".jpg",".jpeg",".png")
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(supported_formats)]

    for filename in sorted(image_files):
        image_path = os.path.join(image_folder, filename)
        print(f"{filename}")

        main(image_path, output_txt)

    exit()
