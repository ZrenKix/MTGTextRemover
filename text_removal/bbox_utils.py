import cv2
import numpy as np

"""
Utilities for working with bounding boxes in text removal operations.
"""

def boxes_are_close(boxA, boxB, threshold):
    """
    Determines whether two bounding boxes either overlap or are within
    the specified threshold distance.
    boxA and boxB are tuples (x1, y1, x2, y2).
    Returns True if they overlap or the gap is within threshold.
    """
    x1a, y1a, x2a, y2a = boxA
    x1b, y1b, x2b, y2b = boxB
    overlapX = not (x2a < x1b or x2b < x1a)
    overlapY = not (y2a < y1b or y2b < y1a)
    if overlapX and overlapY:
        return True
    horiz_dist = 0
    vert_dist = 0
    if x2a < x1b:
        horiz_dist = x1b - x2a
    elif x2b < x1a:
        horiz_dist = x1a - x2b
    if y2a < y1b:
        vert_dist = y1b - y2a
    elif y2b < y1a:
        vert_dist = y1a - y2b
    return max(horiz_dist, vert_dist) <= threshold

def combine_boxes_close(boxes, threshold, debug=False):
    """
    Merges bounding boxes in a list if they overlap or are within
    the specified threshold distance. Returns a new list of merged boxes.
    """
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1a, y1a, x2a, y2a = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                x1b, y1b, x2b, y2b = boxes[j]
                if boxes_are_close((x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b), threshold):
                    nx1 = min(x1a, x1b)
                    ny1 = min(y1a, y1b)
                    nx2 = max(x2a, x2b)
                    ny2 = max(y2a, y2b)
                    used[j] = True
                    x1a, y1a, x2a, y2a = nx1, ny1, nx2, ny2
                    merged = True
            used[i] = True
            new_boxes.append((x1a, y1a, x2a, y2a))
        boxes = new_boxes
    return boxes