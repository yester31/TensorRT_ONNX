# by yhpark 2023-02-19
from utils import *
from torchsummary import summary

# 0. setting parameters
half = False # f16
iteration = 1000

def main():

    # 1. model
    net = load_resnet18()               # resnet18 model load
    net = net.eval()                    # set evaluation mode
    device = device_check()             # device check
    net = net.to(device)                # to gpu
    if half:
        net.half()                      # to FP16

    print(f"model: {net}")              # print model structure
    summary(net, (3, 224, 224))         # print output shape & total parameter sizes for given input size

    # 2. input
    img = cv2.imread('./../data/panda0.jpg') # image file load
    img_ = preprocess(img, half, device)

    # 3. inference
    out = net(img_)                           # except first inference
    torch.cuda.synchronize()
    dur_time = 0
    for i in range(iteration):
        begin = time.time()
        out = net(img_)
        torch.cuda.synchronize()
        dur = time.time() - begin
        dur_time += dur
        #print(f'{i} dur time : {dur}')

    # 4. results
    print(f'{iteration}th iteration time : {dur_time} [sec]')
    print(f'Average fps : {1/(dur_time/iteration)} [fps]')
    print(f'Average inference time : {(dur_time/iteration)*1000} [msec]')
    max_tensor = out.max(dim=1)
    max_value = max_tensor[0].cpu().data.numpy()[0]
    max_index = max_tensor[1].cpu().data.numpy()[0]
    print(f'Resnet18 max index : {max_index} , value : {max_value}, class name : {class_name[max_index]}')

if __name__ == '__main__':
    main()

# 100th iteration time : 0.36880016326904297 [sec]
# Average fps : 271.14955458153935 [fps]
# Average inference time : 3.6880016326904297 [msec]
# Resnet18 max index : 388 , value : 13.548330307006836, class name : giant panda panda panda bear coon bear Ailuropoda melanoleuca