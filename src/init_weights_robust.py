import torch
import argparse
import os
# 尝试导入 safetensors，如果没装会自动回退到 bin
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_list', type=str)
    parser.add_argument('--output_path', type=str, default="result/merging_weights.pth")
    args = parser.parse_args()

    # 1. 解析 adapter 列表
    try:
        adapter_paths = eval(args.adapter_list)
        num_adapters = len(adapter_paths)
        first_adapter_path = adapter_paths[0]
    except:
        print("Error parsing adapter_list.")
        return

    print(f"[Init] Reference adapter: {first_adapter_path}")

    # 2. 寻找 adapter 文件 (支持 .safetensors 和 .bin)
    # 我们只需要读取其中一个 adapter 的键名(Keys)即可
    file_path_safe = os.path.join(first_adapter_path, "adapter_model.safetensors")
    file_path_bin = os.path.join(first_adapter_path, "adapter_model.bin")
    
    keys = []
    
    if os.path.exists(file_path_safe):
        if load_safetensors is None:
             raise ImportError("Found .safetensors file but 'safetensors' library is not installed. Please pip install safetensors.")
        print(f"[Init] Loading keys from: {file_path_safe}")
        state_dict = load_safetensors(file_path_safe)
        keys = list(state_dict.keys())
    elif os.path.exists(file_path_bin):
        print(f"[Init] Loading keys from: {file_path_bin}")
        state_dict = torch.load(file_path_bin, map_location="cpu")
        keys = list(state_dict.keys())
    else:
        # 如果找不到 adapter 文件，尝试去 checkpoint 子目录找
        print(f"[Warning] No adapter model found in {first_adapter_path}. Checking subdirectories...")
        found = False
        for root, dirs, files in os.walk(first_adapter_path):
            for file in files:
                if file == "adapter_model.safetensors":
                    full_path = os.path.join(root, file)
                    print(f"[Init] Found in subdir: {full_path}")
                    state_dict = load_safetensors(full_path)
                    keys = list(state_dict.keys())
                    found = True
                    break
            if found: break
        
        if not found:
            raise FileNotFoundError(f"Could not find adapter_model.safetensors or .bin in {first_adapter_path}")

    # 3. 构造权重字典
    # 结构: { "param_name": tensor([1.0, 1.0, ...]) }
    weights_dict = {}
    for k in keys:
        # 创建一个全 1 的向量，长度等于 adapter 的数量
        weights_dict[k] = torch.ones(num_adapters, dtype=torch.float32)

    # 4. 保存
    dir_name = os.path.dirname(args.output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    torch.save(weights_dict, args.output_path)
    print(f"[Init] Successfully created weights dict with {len(keys)} parameters.")
    print(f"[Init] Saved to: {args.output_path}")

if __name__ == "__main__":
    main()