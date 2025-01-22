import cv2

"""
Contains default configuration constants for text removal operations.
"""

DEFAULT_PHRASES = [
    "Playtest Card — Not for sale",
    "WillieTanner Proxy e NOT FOR SALE",
    "Proxy e NOT FOR SALE",
    "« Not for Sale ¢ TTP Proxy ¢",
    "Not for Sale ¢ TTP Proxy ¢",
    "Custom Proxy « NOT FOR SALE", 
    "Custom Proxy e NOT FOR SALE",
    "Playtest Card - Not for sale", 
    "mpcautofill.com", 
    "not for sanctioned play", 
    "proxy by deelight", 
    "Not for sale!", 
    "Not for sale", 
    "Playtest Card-", 
    "Playtest Card -", 
    "Playtest Card", 
    "Custom Proxy", 
    "TTP Proxy",
    "WillieTanner Proxy Not for sale",
    "WillieTanner Proxy",
    "PsilosX Proxy",
    "Custom Proxy",
    "Proxy"
]

DEFAULT_INPAINT_RADIUS = 3
DEFAULT_INPAINT_METHOD = cv2.INPAINT_TELEA
DEFAULT_PAD_WIDTH = 8
DEFAULT_PAD_HEIGHT = 0
DEFAULT_COMBINE_THRESHOLD = 50
DEFAULT_DILATE = True
DEFAULT_KERNEL_SIZE = 5
DEFAULT_MAX_WORKERS = 8