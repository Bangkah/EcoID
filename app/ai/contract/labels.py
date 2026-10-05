"""Label set. Order MUST match the model's output (class index 0..4).

Training script and export script must use this exact order.
"""

CLASSES = (
    "Mangifera indica",    # Mangga
    "Cocos nucifera",      # Kelapa
    "Musa acuminata",      # Pisang
    "Carica papaya",       # Pepaya
    "Manihot esculenta",   # Singkong
)

COMMON_NAMES = {
    "Mangifera indica": "Mangga",
    "Cocos nucifera": "Kelapa",
    "Musa acuminata": "Pisang",
    "Carica papaya": "Pepaya",
    "Manihot esculenta": "Singkong",
}

NUM_CLASSES = len(CLASSES)
