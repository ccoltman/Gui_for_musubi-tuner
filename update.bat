call .\venv\Scripts\Activate
pip install easydict ftfy opencv-python av diffusers voluptuous gradio typing-extensions
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U accelerate bitsandbytes xformers
pip install https://github.com/sdbds/SageAttention-for-windows/releases/download/2.11_torch270+cu128/sageattention-2.1.1+cu128torch2.7.0-cp310-cp310-win_amd64.whl
pip install -U triton-windows^<3.3
install https://huggingface.co/880ga/flash-attn-274post1-cp310-cu128/resolve/main/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
pause
