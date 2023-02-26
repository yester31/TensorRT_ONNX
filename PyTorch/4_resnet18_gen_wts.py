# by yhpark 2023-02-21
# resnet18 weight extractor for tensorrt cpp example
from utils import *
def main():

    # 1. model
    net = load_resnet18()               # resnet18 model load
    net = net.eval()                    # set evaluation mode
    device = device_check()             # device check
    net = net.to(device)                # to gpu

    if os.path.isfile('model/resnet18.wts'):
        print('Already, resnet18.wts file exists.')
    else:
        print('Create resnet18.wts file ...')        # Create resnet18.wts file if it doesn't exist
        f = open("model/resnet18.wts", 'w')
        f.write("{}\n".format(len(net.state_dict().keys())))
        for k, v in net.state_dict().items():
            print('key: ', k)
            print('value: ', v.shape)
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")
        print('Completed resnet18.wts file!')

if __name__ == '__main__':
    main()