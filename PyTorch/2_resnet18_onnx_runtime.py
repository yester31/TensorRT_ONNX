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
    export_model_path = f"./model/resnet18_{device.type}.onnx"

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
                              export_params=True,       # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                              opset_version=opset_ver,  # 모델을 변환할 때 사용할 ONNX 버전
                              do_constant_folding=True, # 최적하시 상수폴딩을 사용할지의 여부
                              input_names =['input'],   # 모델의 입력값을 가리키는 이름
                              output_names=['output'],  # 모델의 출력값을 가리키는 이름
                              dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
                                            'output': {0: 'batch_size'}})
            print("ONNX Model exported at ", export_model_path)

    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)

    assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
    sess_options = onnxruntime.SessionOptions()
    sess_options.optimized_model_filepath = f"./model/opti_resnet18_{device.type}.onnx"
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    ort_net = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # 2. input
    img = cv2.imread('./../data/panda0.jpg') # image file load
    img_ = preprocess(img, half, device)
    ort_inputs = {ort_net.get_inputs()[0].name: to_numpy(img_)}

    # 3. inference
    out = ort_net.run(None, ort_inputs)[0]
    torch.cuda.synchronize()
    dur_time = 0
    for i in range(iteration):
        begin = time.time()
        out = ort_net.run(None, ort_inputs)[0]
        torch.cuda.synchronize()
        dur = time.time() - begin
        dur_time += dur
        #print('{} dur time : {}'.format(i, dur))

    # 4. results
    print(f'{iteration}th iteration time : {dur_time} [sec]')
    print(f'Average fps : {1/(dur_time/iteration)} [fps]')
    print(f'Average inference time : {(dur_time/iteration)*1000} [msec]')
    max_index = np.argmax(out)
    max_value = out[0, max_index]
    print(f'Resnet18 max index : {max_index} , value : {max_value}, class name : {class_name[max_index]}')


if __name__ == '__main__':
    main()

# CPU
# 100th iteration time : 0.3059732913970947 [sec]
# Average fps : 326.8259119722288 [fps]
# Average inference time : 3.0597329139709473 [msec]
# Resnet18 max index : 388 , value : 13.553796768188477, class name : giant panda panda panda bear coon bear Ailuropoda melanoleuca

# GPU
# 100th iteration time : 0.2522242069244385 [sec]
# Average fps : 396.4726511359716 [fps]
# Average inference time : 2.5222420692443848 [msec]
# Resnet18 max index : 388 , value : 13.553470611572266, class name : giant panda panda panda bear coon bear Ailuropoda melanoleuca