"""Shared constants for the captcha CRNN model."""

# Output classes, in the order the model was trained with.
# Index 0..9 digits, 10..35 uppercase letters, 36 the CTC blank symbol.
LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
NUM_CLASSES = len(LABELS)          # 37
BLANK_INDEX = NUM_CLASSES - 1      # 36, CTC blank

# Model geometry.
TIME_STEPS = 20                    # recurrent output length (max decodable characters is TIME_STEPS - 1)
INPUT_HEIGHT = 60
INPUT_WIDTH = 160
INPUT_CHANNELS = 3

# Default model file names (bundled with the package).
MODEL_FILENAME = "captcha_crnn.onnx"
WEIGHTS_FILENAME = "captcha_crnn_weights.npz"
