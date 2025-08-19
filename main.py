import argparse
import numpy as np
import imutils
import cv2
import os
import math

def perspective_transform(image, gray, template, threshold=0.8):
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, gray_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    w, h = gray_template.shape[::-1]
    centers = [(int(pt[0] + w // 2), int(pt[1] + h // 2)) for pt in zip(*loc[::-1])]
    
    coords = np.array(centers)
    sum_xy = coords[:, 0] + coords[:, 1]
    diff_xy = coords[:, 0] - coords[:, 1]

    tl = coords[np.argmin(sum_xy)]
    tr = coords[np.argmax(diff_xy)]
    bl = coords[np.argmin(diff_xy)]
    br = np.array([tr[0], bl[1]])
    #br = coords[np.argmax(sum_xy)]
    corner_pts = [tl,tr,br,bl]

    pts_src = np.array([tl,tr,br,bl], dtype='float32')
    width = max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl))
    height = max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr))
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
    
    warped = cv2.warpPerspective(image, M_offset, (new_w, new_h))
    return warped, corner_pts 


def find_alignment_bar_y_coords(gray, bound):
    alignment_bar = gray[:, 0:bound]
    thresh = cv2.adaptiveThreshold(alignment_bar, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    alignment_boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w/float(h)
        if 1.5 < aspect_ratio < 10 and 1 < h < 20:
            alignment_boxes.append((x,y,w,h))

    y_coords = sorted([y + h//2 for (_,y,_,h) in alignment_boxes])
    threshold = 5
    y_coords = group_bubbles(y_coords, threshold)
    return y_coords


def dedup_keep_largest(bubbles, threshold=10):
    out = []
    for x,y,r in sorted(bubbles, key=lambda b: b[2], reverse=True):
        if not any(math.hypot(x-bx, y-by) <= threshold for bx,by,_ in out):
            out.append((x,y,r))
    return out


# 40 daki 0'ı algılıyor baloncuk olarak
def find_bubbles_on_row(image, y_center, row_height=40, min_r=15, max_r=22, circ_thr=0.7):

    h, w = image.shape[:2]
    y1 = max(y_center - row_height//2, 0)
    y2 = min(y_center + row_height//2, h)
    
    row_crop = image[y1:y2, :]   
    gray_crop = cv2.cvtColor(row_crop, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_crop, (3,3), 0)
    
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=80, param2=20, minRadius=min_r, maxRadius=max_r)
    bubbles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x,y,r) in circles:
            bubbles.append((x, y+y1, r))

    return sorted(bubbles, key=lambda b: b[0])


def find_columns_xs_grouped(image, y_coords, start_index, count, group_size, start=None, end=None, threshold=10):
    x_coords = set()
    for i in range(start_index, start_index+count):
        bubbles = find_bubbles_on_row(image, y_coords[i])
        for (x,y,r) in bubbles:
            x_coords.add(x)
            #cv2.circle(image, (x,y), r, (0, 0, 255), 2)

    grouped = group_bubbles(x_coords, threshold)
    if end is None:
        end = len(grouped)
    if start is None:
        start = 0
    grouped_sorted = sorted(grouped)[start:end]
    columns = [grouped_sorted[i:i+group_size] for i in range(0, len(grouped_sorted), group_size)]
    print(f"Columns found: {columns}")
    return columns    


def group_bubbles(coords, threshold=10):
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


def crop_column_region(image, columns, padding=20):
    x_min = max(min(columns)-padding, 0)
    x_max = min(max(columns)+padding, image.shape[1])
    cropped = image[:, x_min:x_max]
    return cropped, x_min


def circular_roi(gray, x, y, r):
    h, w = gray.shape[:2]
    y1, y2 = max(0, y-r), min(h, y+r)
    x1, x2 = max(0, x-r), min(w, x+r)

    sub = gray[y1:y2, x1:x2].copy()
    mask = np.zeros_like(sub, dtype=np.uint8)

    cy, cx = y - y1, x - x1
    cv2.circle(mask, (cx, cy), r, 255, -1)     # dolu daire
    circ = cv2.bitwise_and(sub, sub, mask=mask)
    outside_mask = cv2.bitwise_not(mask)
    circ[outside_mask == 255] = 255
    return circ


def check_answers(image, gray, columns, y_coords, threshold):
    ref_r = None
    correct = []
    wrong = []
    #print(columns)
    for index, column in enumerate(columns):
        #print(column)
        question_count = SUBJECTS[index]["question_count"]
        answer_key = SUBJECTS[index]["answer_key"]
        cropped_column, offset = crop_column_region(image, column)
        for i in range(question_count):
            row_index = first_question_index + i
            bubbles = find_bubbles_on_row(cropped_column, y_coords[row_index])
            bubbles = [(int(x+offset), int(y), int(r)) for (x,y,r) in bubbles]
            
            r_val = 0
            if bubbles and len(bubbles) > 0:
                rs = [b[2] for b in bubbles]
                r_val = int(np.median(rs))

            if ref_r is None:
                ref_r = bubbles[0][2]

            # detected_xs = [b[0] for b in bubbles]
            # answer_index = None
            # for i, ref_x in enumerate(column):
            #     matched = any(abs(ref_x-detected_x) <= threshold for detected_x in detected_xs)
            #     if not matched:
            #         answer_index = i
            #         break

            correct_answer_index = row_index-first_question_index
            # if answer_index is not None:
            #     if answer_index == answer_key[correct_answer_index]:
            #         correct.append((index, correct_answer_index+1))
            #         SUBJECTS[index]["correct"]+=1
            #         cv2.circle(image, (column[answer_index], y_coords[row_index]), bubbles[0][2], (0, 255, 0), 2)
            #     else:
            #         SUBJECTS[index]["wrong"]+=1
            #         wrong.append((index, correct_answer_index+1))
            #         cv2.circle(image, (column[answer_index], y_coords[row_index]), bubbles[0][2], (0, 0, 255), 2)
            # else:
            marked = []
            for i, (x,y,r) in enumerate(bubbles):
                print("x:", str(x), "y:", str(y), "r:", str(r))
                roi = circular_roi(gray, x, y, r)
                cv2.imshow("roi", roi)
                cv2.waitKey(0)
                mean = np.mean(roi)
                if mean < bubble_threshold:
                    marked.append(i)
            if len(marked) == 0:
                a=1 #temporary
            elif len(marked) > 1:
                for j in marked:
                    (x, y, r) = bubbles[j]
                    cv2.circle(image, (x, y), r, (0, 255, 255), 2)
            else:
                if marked[0] == answer_key[correct_answer_index]:
                    correct.append((index, correct_answer_index+1))
                    SUBJECTS[index]["correct"]+=1
                    cv2.circle(image, (bubbles[marked[0]][0],bubbles[marked[0]][1]), bubbles[marked[0]][2], (0, 255, 0), 2)
                else:
                    SUBJECTS[index]["wrong"]+=1
                    wrong.append((index, correct_answer_index+1))
                    cv2.circle(image, (bubbles[marked[0]][0],bubbles[marked[0]][1]), bubbles[marked[0]][2], (0, 0, 255), 2)
    return image, correct, wrong, ref_r


def check_column_based_marks(image, gray, columns, y_coords, type_dict, start_row_index, row_count, mark_count, r, threshold=10):
    key = ""
    key_length = 0
    for col_index, column in enumerate(columns):
        x = column[0]
        found_digit = None
        for i in range(row_count):
            y_index = start_row_index + i
            y = y_coords[y_index]

            roi = gray[y - threshold:y + threshold, x - threshold:x + threshold]
            mean = np.mean(roi)
            if mean < bubble_threshold:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                if found_digit is not None:
                    found_digit = "*"
                    break
                if col_index == 0 and type_dict is ID_LETTER:
                    found_digit = type_dict.get(i, "-")
                elif row_count == 1 and type_dict is not ID_LETTER:
                    found_digit = type_dict.get(col_index, "-")
                else:
                    found_digit = str(i)
                key_length+=1
                key += found_digit
        if key_length == mark_count:
            break
    return image, key


def resize_image(image):
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
    image = cv2.imread(image_path)   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread("images/ref_circle.png")
    image, corner_pts = perspective_transform(image, gray, template, threshold=0.8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    y_coords = find_alignment_bar_y_coords(gray, corner_pts[0][0]-25)
    print(f"Y coords: {y_coords}")
    threshold = 10 # Threshold for bubbles' x positions.
    answer_columns = find_columns_xs_grouped(image, y_coords, first_question_index, answer_columns_row_count, bubble_count_per_question, None, None, threshold)
    print(f"Answer columns: {answer_columns}")
    image, correct, wrong, ref_r = check_answers(image, gray, answer_columns, y_coords, threshold)    
    id_columns = find_columns_xs_grouped(image, y_coords, first_id_index, id_row_count, 1, None, 10, threshold)
    print(f"ID columns: {id_columns}")
    image, id = check_column_based_marks(image, gray, id_columns, y_coords, ID_LETTER, first_id_index, id_row_count, id_length, ref_r, threshold)
    print(f"ID: {id}")
    book_type_columns = find_columns_xs_grouped(image, y_coords, book_type_index, 1, 1, None, None, threshold=20)
    print(f"Book type columns: {book_type_columns}")
    image, book_type = check_column_based_marks(image, gray, book_type_columns, y_coords, BOOK_TYPE, book_type_index, 1, 1, ref_r, threshold)
    print(f"Book type: {book_type}")


    with open(output_txt_path, "a", encoding="utf-8") as f:
        f.write("----------------------------------\n")
        f.write(f"{os.path.basename(image_path)}\n")
        f.write(f"Correct answer count of {SUBJECTS[0]["name"]}: {SUBJECTS[0]["correct"]}\n")
        f.write(f"Wrong answer count of {SUBJECTS[0]["name"]}: {SUBJECTS[0]["wrong"]}\n")
        corrects = [b for a, b in correct if a == 0]
        f.write(f"Corrects : {corrects}\n")
        wrongs = [b for a, b in wrong if a == 0]
        f.write(f"Wrongs : {wrongs}\n")
        f.write("ID: " + id + "\n")
        f.write("Book type: " + book_type + "\n")

    image = resize_image(image)
    cv2.imshow("image", image)
    cv2.waitKey(0)

    # print(f"Correct answer count of {SUBJECTS[0]["name"]}: {SUBJECTS[0]["correct"]}")
    # print(f"Wrong answer count of {SUBJECTS[0]["name"]}: {SUBJECTS[0]["wrong"]}")
    # corrects = [b for a, b in correct if a == 0]
    # print(f"Corrects : {corrects}")
    # wrongs = [b for a, b in wrong if a == 0]
    # print(f"Wrongs : {wrongs}")

    # print(f"Correct answer count of {SUBJECTS[1]["name"]}: {SUBJECTS[1]["correct"]}")
    # print(f"Wrong answer count of {SUBJECTS[1]["name"]}: {SUBJECTS[1]["wrong"]}")
    # corrects = [b for a, b in correct if a == 1]
    # print(f"Corrects : {corrects}")
    # wrongs = [b for a, b in wrong if a == 1]
    # print(f"Wrongs : {wrongs}")
    # print(f"Correct answer count of {SUBJECTS[2]["name"]}: {SUBJECTS[2]["correct"]}")
    # print(f"Wrong answer count of {SUBJECTS[2]["name"]}: {SUBJECTS[2]["wrong"]}")
    # corrects = [b for a, b in correct if a == 2]
    # print(f"Corrects : {corrects}")
    # wrongs = [b for a, b in wrong if a == 2]
    # print(f"Wrongs : {wrongs}")
    #
    # print("ID: " + id)
    # print("Book type: " + book_type)

    SUBJECTS[0]["correct"] = 0
    SUBJECTS[0]["wrong"] = 0
    SUBJECTS[1]["correct"] = 0
    SUBJECTS[1]["wrong"] = 0
    SUBJECTS[2]["correct"] = 0
    SUBJECTS[2]["wrong"] = 0



if __name__ == "__main__":
    answer_columns_row_count = 40
    bubble_threshold = 170
    first_question_index = 17
    bubble_count_per_question = 5
    first_id_index = 4
    id_row_count = 10
    book_type_index = 3
    id_length = 10
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
    ID_LETTER = {0:"B", 1:"G", 2:"D", 3:"Y", 4:"Y", 5:"U", 6:"E", 7:"T", 8:"M"}
    BOOK_TYPE = {0:"A", 1:"B", 2:"C", 3:"D"}

    image_folder = "/home/can/Desktop/omr/images/3849200"
    output_txt = "results.txt"

    supported_formats = (".jpg",".jpeg",".png")
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(supported_formats)]

    for filename in sorted(image_files):
        image_path = os.path.join(image_folder, filename)
        print(f"{filename} processing...")

        main(image_path, output_txt)

    exit()

