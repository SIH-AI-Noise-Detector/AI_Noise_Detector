from rknn.api import RKNN

ONNX_MODEL = "models/dccrn_basic.onnx"
RKNN_MODEL = "models/dccrn_basic.rknn"
rknn = RKNN(verbose=True)

print("Configuring RKNN for RK3588...")

ret = rknn.config(
    target_platform="rk3588"
)

if ret != 0:
    print("ERROR: RKNN configuration failed")
    rknn.release()
    exit(ret)

print("RKNN configured successfully.")

print("Loading ONNX model...")

ret = rknn.load_onnx(
    model=ONNX_MODEL
)

if ret != 0:
    print("ERROR: Failed to load ONNX model")
    rknn.release()
    exit(ret)

print("ONNX model loaded successfully.")

print("Building RKNN model...")

ret = rknn.build(
    do_quantization=False
)

if ret != 0:
    print("ERROR: RKNN build failed")
    rknn.release()
    exit(ret)

print("RKNN build successful.")

print("Exporting RKNN model...")

ret = rknn.export_rknn(RKNN_MODEL)

if ret != 0:
    print("ERROR: RKNN export failed")
    rknn.release()
    exit(ret)

print("RKNN model saved to:", RKNN_MODEL)

rknn.release()
