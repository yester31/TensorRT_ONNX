# by yhpark 2023-02-19
from utils import *
import torch.onnx
import onnx
import onnxruntime
import psutil

# 0. setting parameters
half = False # f16
iteration = 100
batch_size = 1  # 임의의 수
enable_overwrite = False

print("pytorch:", torch.__version__)
print("onnxruntime:", onnxruntime.__version__)
print("onnx:", onnx.__version__)

def main():

    device = device_check()  # device check
    #device = torch.device("cpu:0")

    # 모델에 대한 입력값
    export_model_path = f"./model/resnet18_{device.type}_trt.onnx"

    # 모델 변환
    if enable_overwrite or not os.path.exists(export_model_path):
        net = load_resnet18()  # resnet18 model load
        net = net.eval()  # set evaluation mode
        net = net.to(device)  # to gpu
        if half:
            net.half()  # to FP16
        dummy_input = torch.randn(batch_size, 3, 224, 224, requires_grad=True).to(device)
        opset_ver = 17  # ONNX opset version

        with torch.no_grad():
            torch.onnx.export(net,                      # 실행될 모델
                              dummy_input,              # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                              export_model_path,        # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                              opset_version=opset_ver,  # 모델을 변환할 때 사용할 ONNX 버전
                              input_names =['input'],   # 모델의 입력값을 가리키는 이름
                              output_names=['output'])  # 모델의 출력값을 가리키는 이름

            print("ONNX Model exported at ", export_model_path)

    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)

    print("ONNX Model check done!")

if __name__ == '__main__':
    main()