import cv2
import numpy as np
import logging
from .config import DEFAULT_INPAINT_RADIUS, DEFAULT_INPAINT_METHOD
from .bbox_utils import combine_boxes_close
from .tesseract_utils import detect_text

"""
Core logic for removing text from images by detecting specified phrases
via Tesseract OCR, creating bounding boxes, and inpainting.
"""

logger = logging.getLogger(__name__)

def remove_phrases(
    image_path,
    phrases,
    tesseract_cmd=None,
    debug=False,
    pad_width=8,
    pad_height=0,
    inpaint_radius=DEFAULT_INPAINT_RADIUS,
    inpaint_method=DEFAULT_INPAINT_METHOD,
    do_dilate=True,
    dilate_kernel_size=5,
    combine_threshold=50
):
    """
    Removes text from an image file by detecting any of the specified phrases
    using Tesseract OCR, then inpainting those regions. 
    Returns the inpainted image or (inpainted_image, debug_overlay) if debug is True.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Could not read image: %s", image_path)
        return None
    debug_image = image.copy() if debug else None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = detect_text(rgb)
    line_data = group_words_by_line(data)
    boxes_to_remove = collect_boxes_for_phrases(line_data, phrases, pad_width, pad_height, debug)
    if combine_threshold > 0 and boxes_to_remove:
        boxes_to_remove = combine_boxes_close(boxes_to_remove, combine_threshold, debug)
    if not boxes_to_remove:
        logger.info("No text matches found. Returning original image.")
        if debug:
            return image, debug_image
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray, dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes_to_remove:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        if debug_image is not None:
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if do_dilate:
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        logger.debug("Mask dilation applied.")
    logger.debug("Inpainting started.")
    image = cv2.inpaint(image, mask, inpaint_radius, inpaint_method)
    logger.debug("Inpainting finished.")
    if debug:
        return image, debug_image
    return image

def group_words_by_line(data):
    """
    Groups recognized text from Tesseract's output by line number.
    Returns a dictionary mapping line_num to a list of (text, x, y, w, h).
    """
    line_data = {}
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        txt = data["text"][i].strip()
        if not txt:
            continue
        line_num = data["line_num"][i]
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        if line_num not in line_data:
            line_data[line_num] = []
        line_data[line_num].append((txt, x, y, w, h))
    return line_data

def collect_boxes_for_phrases(line_data, phrases, pad_width, pad_height, debug=False):
    """
    Collects bounding boxes for words matching any of the specified phrases,
    returning a list of (x1, y1, x2, y2).
    """
    boxes_to_remove = []
    phrase_lists = []
    for p in phrases:
        p_list = [w.strip() for w in p.split() if w.strip()]
        if p_list:
            phrase_lists.append(p_list)
    for line_num, words_info in line_data.items():
        if debug:
            line_text_joined = " ".join([wi[0] for wi in words_info])
            print(f"[DEBUG] Line {line_num}: '{line_text_joined}'")
        indices_to_remove = set()
        for phrase_list in phrase_lists:
            ph_len = len(phrase_list)
            for start_idx in range(len(words_info) - ph_len + 1):
                window_texts = [words_info[j][0] for j in range(start_idx, start_idx + ph_len)]
                if match_window(window_texts, phrase_list):
                    for j in range(start_idx, start_idx + ph_len):
                        indices_to_remove.add(j)
                    if debug:
                        matched_str = " ".join(window_texts)
                        print(f"    [DEBUG] Matched phrase: '{matched_str}'")
        for idx in indices_to_remove:
            _, x, y, w, h = words_info[idx]
            x1 = max(x - pad_width, 0)
            y1 = max(y - pad_height, 0)
            x2 = x + w + pad_width
            y2 = y + h + pad_height
            boxes_to_remove.append((x1, y1, x2, y2))
    return boxes_to_remove

def match_window(word_list, phrase_list):
    """
    Checks if the words in word_list match phrase_list exactly, ignoring case.
    """
    if len(word_list) != len(phrase_list):
        return False
    for w1, w2 in zip(word_list, phrase_list):
        if w1.lower() != w2.lower():
            return False
    return True