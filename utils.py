import os
import torch

def get_device():
    """
    Returns the best available device (CUDA, MPS, or CPU) along with a name.
    """
    if torch.cuda.is_available():
        return torch.device('cuda'), "CUDA GPU: " + torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            test_tensor = torch.ones(1).to('mps')
            _ = test_tensor + test_tensor
            return torch.device('mps'), "Apple Metal GPU"
        except Exception as e:
            print(f"Warning: MPS initialization failed, falling back to CPU. Error: {str(e)}")
            return torch.device('cpu'), "CPU (MPS fallback)"
    return torch.device('cpu'), "CPU"

def to_device(data, device):
    """
    Recursively moves data (tensor, list, dict) to the target device.
    """
    try:
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        elif isinstance(data, dict):
            return {k: to_device(v, device) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        return data
    except Exception as e:
        print(f"Warning: Failed to move data to device {device}, using CPU. Error: {str(e)}")
        if isinstance(data, torch.Tensor):
            return data.cpu()
        return data

def get_unique_filepath(filepath):
    """
    If the filepath exists, appends a counter to create a unique filepath.
    """
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = f"{base}_{counter}{ext}"
    while os.path.exists(new_filepath):
        counter += 1
        new_filepath = f"{base}_{counter}{ext}"
    return new_filepath