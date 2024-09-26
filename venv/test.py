import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


file_path = 'venv/timestamped.txt'
with open(filepath, 'r', encoding='utf-8') as file: