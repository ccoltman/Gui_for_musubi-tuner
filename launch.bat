chcp 65001
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git pull

venv\Scripts\pip install easydict ftfy opencv-python av diffusers voluptuous gradio typing-extensions
venv\Scripts\pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
venv\Scripts\pip install --upgrade accelerate
venv\Scripts\pip install -U bitsandbytes
venv\Scripts\pip install -U xformers
venv\Scripts\pip install https://github.com/sdbds/SageAttention-for-windows/releases/download/2.11_torch270%2Bcu128/sageattention-2.1.1+cu128torch2.7.0-cp310-cp310-win_amd64.whl
venv\Scripts\pip install -U triton-windows^<3.3
venv\Scripts\pip install https://huggingface.co/880ga/flash-attn-274post1-cp310-cu128/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
venv\Scripts\python.exe .\train_gui.py
pause
