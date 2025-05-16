call .\venv\Scripts\Activate
chcp 65001
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git pull
python .\train_gui.py
pause
