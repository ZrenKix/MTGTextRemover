import pytesseract
from pytesseract import Output

"""
Tesseract utilities for text detection and configuration.
"""

def detect_text(rgb_image):
    """
    Returns OCR data from Tesseract as a dictionary with bounding box
    information for each recognized word.
    """
    return pytesseract.image_to_data(rgb_image, output_type=Output.DICT)

def configure_tesseract(tesseract_cmd=None):
    """
    Sets the Tesseract command if a custom path is provided.
    Otherwise uses the system default.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd