# by yhpark 2023-02-19
# resnet18 onnx model generation for tensorrt cpp example
from utils import *
import torch.onnx
import onnx
import onnxruntime

# 0. setting parameters
half = False  # f16
batch_size = 1
enable_overwrite = False

print("pytorch:", torch.__version__)
print("onnxruntime:", onnxruntime.__version__)
print("onnx:", onnx.__version__)

def main():
    device = device_check()  # device check & define
    # device = torch.device("cpu:0")

    # onnx model path
    export_model_path = f"./model/resnet18_{device.type}_trt.onnx"

    # 1. export onnx model
    if enable_overwrite or not os.path.exists(export_model_path):
        net = load_resnet18()   # resnet18 model load
        net = net.eval()        # set evaluation mode
        net = net.to(device)    # to gpu
        if half:
            net.half()  # to FP16

        dummy_input = torch.randn(batch_size, 3, 224, 224, requires_grad=True).to(device)

        with torch.no_grad():
            torch.onnx.export(net,                      # pytorch model
                              dummy_input,              # model dummy input
                              export_model_path,        # onnx model path
                              opset_version=17,         # the version of the opset
                              input_names=['input'],    # input name
                              output_names=['output'])  # output name

            print("ONNX Model exported at ", export_model_path)

    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX Model check done!")

if __name__ == '__main__':
    main()
