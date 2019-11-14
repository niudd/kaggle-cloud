import subprocess
import re
import os
import torch


def show_gpu_usage():
    info = subprocess.check_output(['nvidia-smi -i 0,1'], shell=True).decode("utf-8")

    pat = re.compile(r'\d+MiB')
    match = pat.findall(info)
    memory_usage0 = match[0]
    memory_usage1 = match[2]

    pat = re.compile(r'\d+%')
    match = pat.findall(info)
    util_usage0 = match[1]
    util_usage1 = match[3]

    gpu0 = {'name': 'gpu0', 'memory_usage': memory_usage0, 'util_usage': util_usage0}
    gpu1 = {'name': 'gpu1', 'memory_usage': memory_usage1, 'util_usage': util_usage1}

    return gpu0, gpu1


def set_n_get_device(device_id, data_device_id="cuda:0"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id#"0"#"0, 1, 2, 3, 4, 5"
    device = torch.device(data_device_id if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #torch.set_num_threads(20)
    return device


if __name__ == "__main__":
    print(show_gpu_usage())