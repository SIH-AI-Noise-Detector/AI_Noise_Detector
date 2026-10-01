from rknn.api import RKNN

RKNN_MODEL = "models/dccrn_basic.rknn"

rknn = RKNN(verbose=True)

print("Loading RKNN model...")

ret = rknn.load_rknn(RKNN_MODEL)

if ret != 0:
    print("ERROR: Failed to load RKNN model")
    rknn.release()
    exit(ret)

print("RKNN model loaded successfully.")

rknn.release()

print("RKNN verification complete.")
