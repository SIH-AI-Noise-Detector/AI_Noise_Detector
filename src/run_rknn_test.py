import numpy as np
from rknn.api import RKNN

RKNN_MODEL = "models/dccrn_basic.rknn"

rknn = RKNN(verbose=False)

print("Loading RKNN model...")

ret = rknn.load_rknn(RKNN_MODEL)

if ret != 0:
    print("ERROR: Failed to load RKNN model")
    rknn.release()
    exit(ret)

print("Model loaded.")

print("Initializing RKNN runtime...")

ret = rknn.init_runtime()

if ret != 0:
    print("ERROR: Runtime initialization failed")
    rknn.release()
    exit(ret)

print("Runtime initialized.")

input_data = np.random.randn(
    1, 2, 257, 251
).astype(np.float32)

print("Running inference...")

outputs = rknn.inference(
    inputs=[input_data]
)

print("Inference successful.")

print("Number of outputs:", len(outputs))
print("Output shape:", outputs[0].shape)
print("Output dtype:", outputs[0].dtype)

rknn.release()

print("RKNN inference test complete.")
