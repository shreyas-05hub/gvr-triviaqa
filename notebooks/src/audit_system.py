# Stage-0 audit: measurements only. No installs, no model/index downloads.
import json, os, platform, shutil, subprocess, sys
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()
print("== python:", sys.version.replace("\n", " "), "| executable:", sys.executable)
print("== os:", sh("cat /etc/os-release | head -2"), "| kernel:", platform.release())
print("== nvidia-smi\n" + sh("nvidia-smi"))
print("== gpu query:", sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv"))
print("== cuda toolkit:", sh("nvcc --version | tail -2"))
print("== cpu:", sh("nproc"), "cores |", sh("lscpu | grep 'Model name'"))
print("== ram\n" + sh("free -h"))
for p in ["/kaggle/working", "/kaggle/temp", "/tmp", "/root", "/", os.path.expanduser("~/.cache")]:
    if os.path.exists(p):
        u = shutil.disk_usage(p); print(f"== disk {p}: free {u.free/1e9:.1f} GB / total {u.total/1e9:.1f} GB")
print("== java:", sh("java -version"), "| JAVA_HOME:", os.environ.get("JAVA_HOME"))
print("== openjdk-21 apt candidate:", sh("apt-cache policy openjdk-21-jdk-headless 2>&1 | head -3"))
print("== uv:", sh("which uv && uv --version"), "| python3.11:", sh("which python3.11"))
print("== preinstalled pkgs:", sh(f"{sys.executable} -m pip list 2>/dev/null | grep -iE '^(torch|transformers|accelerate|bitsandbytes|datasets|pyserini|faiss|numpy) '"))
try:
    import torch
    print("== torch", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"   gpu{i}: {p.name} vram={p.total_memory/1e9:.2f} GB cc={p.major}.{p.minor}")
        x = torch.randn(256, 256, device="cuda", dtype=torch.float16); print("   fp16 matmul ok:", float((x @ x).float().sum()) == float((x @ x).float().sum()))
except Exception as e:
    print("== torch import failed:", e)
drv = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").split("\n")[0]
print("== NOTE: lock pins torch==2.14.1 PyPI wheel (CUDA 13 runtime). CUDA 13 needs a Linux driver >= 580 and"
      " compute capability >= 7.5 (T4 ok; P100 = 6.0 not supported). Measured driver:", drv)
print("== internet:", sh("curl -sI https://huggingface.co | head -1"), "|", sh("curl -sI https://rgw.cs.uwaterloo.ca | head -1"))
