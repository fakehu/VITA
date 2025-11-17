import json
import matplotlib.pyplot as plt
import numpy as np
import torch

def load_jsonl(data_path):
    data_list = []
    with open(data_path, 'r') as f:
        data_lines = f.readlines()
        for line in data_lines:
            data_dict = {}
            data_dict = json.loads(line)
            data_list.append(data_dict)
    print(f"成功加载文件：{data_path}")
    
    return data_list