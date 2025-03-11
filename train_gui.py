import gradio as gr
import subprocess
import sys
import os
import signal
import psutil
from typing import Generator
import toml  # 用于保存和加载设置

#########################
# 1. 全局进程管理
#########################

running_processes = {
    "cache": None,   # 预缓存进程
    "train": None    # 训练进程
}

def terminate_process_tree(proc: subprocess.Popen):
    """
    递归终止指定进程及其所有子进程，适用于加速器或多进程场景。
    """
    if proc is None:
        return
    try:
        parent_pid = proc.pid
        if parent_pid is None:
            return
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"[WARN] terminate_process_tree 出现异常: {e}")

def stop_caching():
    """
    停止当前正在运行的预缓存子进程。
    """
    if running_processes["cache"] is not None:
        proc = running_processes["cache"]
        if proc.poll() is None:
            terminate_process_tree(proc)
            running_processes["cache"] = None
            return "[INFO] 已请求停止预缓存进程（杀掉所有子进程）。\n"
        else:
            return "[WARN] 预缓存进程已经结束，无需停止。\n"
    else:
        return "[WARN] 当前没有正在进行的预缓存进程。\n"

def stop_training():
    """
    停止当前正在运行的训练子进程。
    """
    if running_processes["train"] is not None:
        proc = running_processes["train"]
        if proc.poll() is None:
            terminate_process_tree(proc)
            running_processes["train"] = None
            return "[INFO] 已请求停止训练进程（杀掉所有子进程）。\n"
        else:
            return "[WARN] 训练进程已经结束，无需停止。\n"
    else:
        return "[WARN] 当前没有正在进行的训练进程。\n"

#########################
# 2. 设置保存与加载
#########################

SETTINGS_FILE = "settings.toml"

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = toml.load(f)
                return settings
        except Exception:
            return {}
    else:
        return {}

def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            toml.dump(settings, f)
    except Exception as e:
        print(f"[WARN] 保存 settings.toml 失败: {e}")

#########################
# 3. 处理输入数据集配置路径
#########################

def get_dataset_config(file_path: str, text_path: str) -> str:
    # 对于 toml 文件，优先使用上传文件的路径
    if file_path and os.path.isfile(file_path):
        return file_path
    elif text_path.strip():
        return text_path.strip()
    else:
        return ""

#########################
# 4. Pre-caching
#########################

# Hunyuan 预缓存：toml 文件上传，其它模型文件路径由用户手动输入
def run_cache_commands(
    dataset_config_file: str,
    dataset_config_text: str,
    enable_low_memory: bool,
    skip_existing: bool,
    use_clip: bool,
    clip_model_path: str,
    vae_path: str,
    text_encoder1_path: str,
    text_encoder2_path: str
) -> Generator[str, None, None]:
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    python_executable = "./python_embeded/python.exe"

    # Latent 预缓存命令（--clip 参数仅添加到此处）
    cache_latents_cmd = [
        python_executable, "cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_path,
        "--vae_chunk_size", "32",
        "--vae_tiling"
    ]
    if enable_low_memory:
        cache_latents_cmd.extend(["--vae_spatial_tile_sample_min_size", "128", "--batch_size", "1"])
    if skip_existing:
        cache_latents_cmd.append("--skip_existing")
    if use_clip and clip_model_path.strip():
        cache_latents_cmd.extend(["--clip", clip_model_path.strip()])

    # Text Encoder 输出预缓存命令（不添加 --clip 参数）
    cache_text_encoder_cmd = [
        python_executable, "cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--text_encoder1", text_encoder1_path,
        "--text_encoder2", text_encoder2_path,
        "--batch_size", "16"
    ]
    if enable_low_memory:
        cache_text_encoder_cmd.append("--fp8_llm")

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["cache"] = process
        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
        return_code = process.wait()
        running_processes["cache"] = None
        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    accumulated_main = "\n[INFO] 开始运行 Hunyuan Latent 预缓存 (cache_latents.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_latents_cmd):
        yield content
    accumulated_main += "\n[INFO] Hunyuan Latent 预缓存已完成。\n"
    yield accumulated_main

    accumulated_main += "\n[INFO] 开始运行 Hunyuan Text Encoder 输出预缓存 (cache_text_encoder_outputs.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_text_encoder_cmd):
        yield content
    accumulated_main += "\n[INFO] Hunyuan Text Encoder 输出预缓存已完成。\n"
    yield accumulated_main

    # 保存所有 Hunyuan 预缓存设置
    pre_caching_settings = {
        "pre_caching": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "enable_low_memory": enable_low_memory,
            "skip_existing": skip_existing,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path,
            "vae_path": vae_path,
            "text_encoder1_path": text_encoder1_path,
            "text_encoder2_path": text_encoder2_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(pre_caching_settings)
    save_settings(existing_settings)

# Wan2.1 预缓存：toml 文件上传，其它模型文件路径由用户手动输入
def run_wan_cache_commands(
    dataset_config_file: str,
    dataset_config_text: str,
    enable_low_memory: bool,
    skip_existing: bool,
    use_clip: bool,
    clip_model_path: str,
    vae_path: str,
    t5_path: str
) -> Generator[str, None, None]:
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    python_executable = "./python_embeded/python.exe"

    cache_latents_cmd = [
        python_executable, "wan_cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_path
    ]
    if enable_low_memory:
        cache_latents_cmd.append("--vae_cache_cpu")
    if skip_existing:
        cache_latents_cmd.append("--skip_existing")
    if use_clip and clip_model_path.strip():
        cache_latents_cmd.extend(["--clip", clip_model_path.strip()])

    cache_text_encoder_cmd = [
        python_executable, "wan_cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--t5", t5_path,
        "--batch_size", "16"
    ]
    if enable_low_memory:
        cache_text_encoder_cmd.append("--fp8_t5")

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["cache"] = process
        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
        return_code = process.wait()
        running_processes["cache"] = None
        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    accumulated_main = "\n[INFO] 开始运行 Wan2.1 Latent 预缓存 (wan_cache_latents.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_latents_cmd):
        yield content
    accumulated_main += "\n[INFO] Wan2.1 Latent 预缓存已完成。\n"
    yield accumulated_main

    accumulated_main += "\n[INFO] 开始运行 Wan2.1 Text Encoder 输出预缓存 (wan_cache_text_encoder_outputs.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_text_encoder_cmd):
        yield content
    accumulated_main += "\n[INFO] Wan2.1 Text Encoder 输出预缓存已完成。\n"
    yield accumulated_main

    # 保存 Wan2.1 预缓存设置（注意保存到 "wan_pre_caching" 以便后续加载）
    wan_pre_caching_settings = {
        "wan_pre_caching": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "enable_low_memory": enable_low_memory,
            "skip_existing": skip_existing,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path,
            "vae_path": vae_path,
            "t5_path": t5_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(wan_pre_caching_settings)
    save_settings(existing_settings)

#########################
# 5. Hunyuan 训练函数
#########################

def make_prompt_file(
    prompt_text: str,
    w: int,
    h: int,
    frames: int,
    seed: int,
    steps: int,
    custom_prompt_txt: bool,
    custom_prompt_path: str,
    prompt_file_upload: str = None
) -> str:
    """
    如果上传了 prompt_file.txt，则直接返回该文件路径；
    否则，如果勾选了自定义且输入了路径，则返回该路径；
    否则自动生成默认的 prompt 文件。
    """
    if prompt_file_upload and os.path.isfile(prompt_file_upload):
        return prompt_file_upload
    elif custom_prompt_txt and custom_prompt_path.strip():
        return custom_prompt_path.strip()
    else:
        default_prompt_path = "./prompt_file.txt"
        with open(default_prompt_path, "w", encoding="utf-8") as f:
            f.write("# prompt 1: for generating a cat video\n")
            line = f"{prompt_text} --w {w} --h {h} --f {frames} --d {seed} --s {steps}\n"
            f.write(line)
        return default_prompt_path

def run_training(
    dataset_config_file: str,
    dataset_config_text: str,
    max_train_epochs: int,
    learning_rate: str,
    network_dim: int,
    network_alpha: int,
    gradient_accumulation_steps: int,
    enable_low_vram: bool,
    blocks_to_swap: int,
    output_dir: str,
    output_name: str,
    save_every_n_epochs: int,
    use_network_weights: bool,
    network_weights_path: str,
    use_clip: bool,
    clip_model_path: str,
    dit_weights_path: str,
    generate_samples: bool,
    sample_every_n_epochs: int,
    sample_every_n_steps: int,
    sample_prompt_text: str,
    sample_w: int,
    sample_h: int,
    sample_frames: int,
    sample_seed: int,
    sample_steps: int,
    custom_prompt_txt: bool,
    custom_prompt_path: str,
    prompt_file_upload: str,
    sample_vae_path: str,
    sample_text_encoder1_path: str,
    sample_text_encoder2_path: str
) -> Generator[str, None, None]:
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    python_executable = "./python_embeded/python.exe"

    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        "hv_train_network.py",
        "--dit", dit_weights_path,
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", "bf16",
        "--fp8_base",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", learning_rate,
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module=networks.lora",
        f"--network_dim={network_dim}",
        f"--network_alpha={network_alpha}",
        "--timestep_sampling", "sigmoid",
        "--discrete_flow_shift", "1.0",
        "--max_train_epochs", str(max_train_epochs),
        "--seed", "42",
        "--output_dir", output_dir,
        "--output_name", output_name,
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        "--logging_dir", "./log",
        "--log_with", "tensorboard",
        "--save_every_n_epochs", str(save_every_n_epochs)
    ]
    if enable_low_vram:
        command.extend(["--blocks_to_swap", str(blocks_to_swap)])
    if use_network_weights and network_weights_path.strip():
        command.extend(["--network_weights", network_weights_path.strip()])
    if use_clip and clip_model_path.strip():
        command.extend(["--clip", clip_model_path.strip()])
    if generate_samples:
        prompt_file_final = make_prompt_file(
            prompt_text=sample_prompt_text,
            w=sample_w,
            h=sample_h,
            frames=sample_frames,
            seed=sample_seed,
            steps=sample_steps,
            custom_prompt_txt=custom_prompt_txt,
            custom_prompt_path=custom_prompt_path,
            prompt_file_upload=prompt_file_upload
        )
        command.extend([
            "--sample_prompts", prompt_file_final,
            "--sample_every_n_epochs", str(sample_every_n_epochs),
            "--sample_every_n_steps", str(sample_every_n_steps),
            "--sample_at_first",
            "--vae", sample_vae_path,
            "--text_encoder1", sample_text_encoder1_path,
            "--text_encoder2", sample_text_encoder2_path,
            "--fp8_llm"
        ])

    current_settings = {
        "training": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "max_train_epochs": max_train_epochs,
            "learning_rate": learning_rate,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "enable_low_vram": enable_low_vram,
            "blocks_to_swap": blocks_to_swap,
            "output_dir": output_dir,
            "output_name": output_name,
            "save_every_n_epochs": save_every_n_epochs,
            "use_network_weights": use_network_weights,
            "network_weights_path": network_weights_path,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path,
            "dit_weights_path": dit_weights_path,
            "generate_samples": generate_samples,
            "sample_every_n_epochs": sample_every_n_epochs,
            "sample_every_n_steps": sample_every_n_steps,
            "sample_prompt_text": sample_prompt_text,
            "sample_w": sample_w,
            "sample_h": sample_h,
            "sample_frames": sample_frames,
            "sample_seed": sample_seed,
            "sample_steps": sample_steps,
            "custom_prompt_txt": custom_prompt_txt,
            "custom_prompt_path": custom_prompt_path,
            "prompt_file_upload": prompt_file_upload,
            "sample_vae_path": sample_vae_path,
            "sample_text_encoder1_path": sample_text_encoder1_path,
            "sample_text_encoder2_path": sample_text_encoder2_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(current_settings)
    save_settings(existing_settings)

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["train"] = process
        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
        return_code = process.wait()
        running_processes["train"] = None
        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    start_message = "[INFO] 开始运行训练命令...\n\n"
    yield start_message
    for content in run_and_stream_output(command):
        yield content
    yield "\n[INFO] 训练命令执行完成。\n"

#########################
# 6. Wan2.1 训练函数
#########################

def run_wan_training(
    dataset_config_file: str,
    dataset_config_text: str,
    task: str,
    dit_weights_path: str,
    max_train_epochs: int,
    learning_rate: str,
    network_dim: int,
    enable_low_vram: bool,
    blocks_to_swap: int,
    output_dir: str,
    output_name: str,
    save_every_n_epochs: int,
    use_network_weights: bool,
    network_weights_path: str,
    use_clip: bool,
    clip_model_path: str,
    timestep_sampling: str,
    discrete_flow_shift: float,
    generate_samples: bool,
    sample_every_n_epochs: int,
    sample_every_n_steps: int,
    sample_prompt_text: str,
    sample_w: int,
    sample_h: int,
    sample_frames: int,
    sample_seed: int,
    sample_steps: int,
    custom_prompt_txt: bool,
    custom_prompt_path: str,
    prompt_file_upload: str,
    sample_vae_path: str,
    sample_t5_path: str
) -> Generator[str, None, None]:
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    python_executable = "./python_embeded/python.exe"

    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        "wan_train_network.py",
        "--task", task,
        "--dit", dit_weights_path,
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", "bf16",
        "--fp8_base",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", learning_rate,
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_wan",
        "--network_dim", str(network_dim),
        "--timestep_sampling", timestep_sampling,
        "--discrete_flow_shift", str(discrete_flow_shift),
        "--max_train_epochs", str(max_train_epochs),
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--seed", "42",
        "--output_dir", output_dir,
        "--output_name", output_name
    ]
    if enable_low_vram:
        command.extend(["--blocks_to_swap", str(blocks_to_swap)])
    if use_network_weights and network_weights_path.strip():
        command.extend(["--network_weights", network_weights_path.strip()])
    if use_clip and clip_model_path.strip():
        command.extend(["--clip", clip_model_path.strip()])
    if generate_samples:
        prompt_file_final = make_prompt_file(
            prompt_text=sample_prompt_text,
            w=sample_w,
            h=sample_h,
            frames=sample_frames,
            seed=sample_seed,
            steps=sample_steps,
            custom_prompt_txt=custom_prompt_txt,
            custom_prompt_path=custom_prompt_path,
            prompt_file_upload=prompt_file_upload
        )
        command.extend([
            "--sample_prompts", prompt_file_final,
            "--sample_every_n_epochs", str(sample_every_n_epochs),
            "--sample_every_n_steps", str(sample_every_n_steps),
            "--sample_at_first",
            "--vae", sample_vae_path,
            "--t5", sample_t5_path,
            "--fp8_llm"
        ])

    current_settings = {
        "wan_training": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "task": task,
            "dit_weights_path": dit_weights_path,
            "max_train_epochs": max_train_epochs,
            "learning_rate": learning_rate,
            "network_dim": network_dim,
            "enable_low_vram": enable_low_vram,
            "blocks_to_swap": blocks_to_swap,
            "output_dir": output_dir,
            "output_name": output_name,
            "save_every_n_epochs": save_every_n_epochs,
            "use_network_weights": use_network_weights,
            "network_weights_path": network_weights_path,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path,
            "timestep_sampling": timestep_sampling,
            "discrete_flow_shift": discrete_flow_shift,
            "generate_samples": generate_samples,
            "sample_every_n_epochs": sample_every_n_epochs,
            "sample_every_n_steps": sample_every_n_steps,
            "sample_prompt_text": sample_prompt_text,
            "sample_w": sample_w,
            "sample_h": sample_h,
            "sample_frames": sample_frames,
            "sample_seed": sample_seed,
            "sample_steps": sample_steps,
            "custom_prompt_txt": custom_prompt_txt,
            "custom_prompt_path": custom_prompt_path,
            "prompt_file_upload": prompt_file_upload,
            "sample_vae_path": sample_vae_path,
            "sample_t5_path": sample_t5_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(current_settings)
    save_settings(existing_settings)

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["train"] = process
        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
        return_code = process.wait()
        running_processes["train"] = None
        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    start_message = "[INFO] 开始运行 Wan2.1 训练命令...\n\n"
    yield start_message
    for content in run_and_stream_output(command):
        yield content
    yield "\n[INFO] Wan2.1 训练命令执行完成。\n"

#########################
# 7. LoRA Conversion
#########################

def run_lora_conversion(lora_file_path: str, output_dir: str) -> Generator[str, None, None]:
    if not lora_file_path.strip() or not os.path.isfile(lora_file_path.strip()):
        yield "[ERROR] 未选择有效的 LoRA 文件路径\n"
        return
    python_executable = "./python_embeded/python.exe"
    in_path = lora_file_path.strip()
    basename = os.path.basename(in_path)
    filename_no_ext, ext = os.path.splitext(basename)
    out_name = f"{filename_no_ext}_converted{ext}"
    if not output_dir.strip():
        output_dir = "."
    out_path = os.path.join(output_dir.strip(), out_name)
    command = [
        python_executable, "convert_lora.py",
        "--input", in_path,
        "--output", out_path,
        "--target", "other"
    ]
    accumulated = ""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="", flush=True)
        accumulated += line
        yield accumulated
    return_code = process.wait()
    if return_code != 0:
        error_msg = f"\n[ERROR] convert_lora.py 运行失败，返回码: {return_code}\n"
        accumulated += error_msg
        yield accumulated
    else:
        msg = f"\n[INFO] LoRA 转换完成！输出文件: {out_path}\n"
        accumulated += msg
        yield accumulated

#########################
# 8. 构建 Gradio UI
#########################

# 注意：toml 文件上传功能保留，其它模型文件及 prompt_file.txt 上传由用户手动输入或上传（上传的 prompt_file.txt 会优先使用）
settings = load_settings()
pre_caching_settings = settings.get("pre_caching", {})
# 新增：加载 Wan2.1 预缓存的设置
wan_pre_caching_settings = settings.get("wan_pre_caching", {})
training_settings = settings.get("training", {})
wan_training_settings = settings.get("wan_training", {})

with gr.Blocks() as demo:
    gr.Markdown("# AI Software Musubi Tuner Gui (双语支持 中/英)")

    ########################################
    # (1) Pre-caching / 预缓存 页面
    ########################################
    with gr.Tab("Pre-caching / 预缓存"):
        with gr.Tabs():
            # Hunyuan 预缓存子标签
            with gr.Tab("Hunyuan Pre-caching / Hunyuan 预缓存"):
                gr.Markdown("## Hunyuan Latent 和 Text Encoder 输出预缓存（toml文件上传，其它文件路径手动输入）")
                with gr.Row():
                    dataset_config_file = gr.File(label="上传 dataset_config (toml)", file_count="single", file_types=[".toml"], type="filepath")
                    dataset_config_text = gr.Textbox(label="或手动输入 toml 路径", placeholder="例如：K:/ai_software/config.toml", value=pre_caching_settings.get("dataset_config_text", ""))
                enable_low_memory = gr.Checkbox(label="启用低显存模式", value=pre_caching_settings.get("enable_low_memory", False))
                skip_existing = gr.Checkbox(label="跳过已存在的 Cache 文件 (--skip_existing)", value=pre_caching_settings.get("skip_existing", False))
                with gr.Row():
                    vae_path = gr.Textbox(label="Hunyuan VAE 文件路径", placeholder="例如：K:/models/hunyuan/vae.pth", value=pre_caching_settings.get("vae_path", ""))
                    text_encoder1_path = gr.Textbox(label="Text Encoder 1 路径", placeholder="例如：K:/models/hunyuan/text_encoder1.pth", value=pre_caching_settings.get("text_encoder1_path", ""))
                    text_encoder2_path = gr.Textbox(label="Text Encoder 2 路径", placeholder="例如：K:/models/hunyuan/text_encoder2.pth", value=pre_caching_settings.get("text_encoder2_path", ""))
                with gr.Row():
                    use_clip_checkbox = gr.Checkbox(label="使用 CLIP 模型 (--clip)", value=pre_caching_settings.get("use_clip", False))
                    clip_model_path = gr.Textbox(label="Hunyuan CLIP 模型文件路径", placeholder="例如：K:/models/hunyuan/clip.pth", visible=False, value=pre_caching_settings.get("clip_model_path", ""))
                def toggle_clip_hunyuan(checked):
                    return gr.update(visible=checked)
                use_clip_checkbox.change(toggle_clip_hunyuan, inputs=use_clip_checkbox, outputs=clip_model_path)
                with gr.Row():
                    run_cache_button = gr.Button("运行 Hunyuan 预缓存")
                    stop_cache_button = gr.Button("停止 Hunyuan 预缓存")
                cache_output = gr.Textbox(label="Hunyuan 预缓存输出", lines=20, interactive=False)
                run_cache_button.click(
                    fn=run_cache_commands,
                    inputs=[dataset_config_file, dataset_config_text, enable_low_memory, skip_existing,
                            use_clip_checkbox, clip_model_path, vae_path, text_encoder1_path, text_encoder2_path],
                    outputs=cache_output
                )
                stop_cache_button.click(fn=stop_caching, inputs=None, outputs=cache_output)
            # Wan2.1 预缓存子标签
            with gr.Tab("Wan2.1 Pre-caching / Wan2.1 预缓存"):
                gr.Markdown("## Wan2.1 Latent 和 Text Encoder 输出预缓存（toml文件上传，其它文件路径手动输入）")
                with gr.Row():
                    dataset_config_file_wan = gr.File(label="上传 dataset_config (toml)", file_count="single", file_types=[".toml"], type="filepath")
                    dataset_config_text_wan = gr.Textbox(label="或手动输入 toml 路径", placeholder="例如：K:/ai_software/config.toml", value=wan_pre_caching_settings.get("dataset_config_text", ""))
                enable_low_memory_wan = gr.Checkbox(label="启用低显存模式", value=wan_pre_caching_settings.get("enable_low_memory", False))
                skip_existing_wan = gr.Checkbox(label="跳过已存在的 Cache 文件 (--skip_existing)", value=wan_pre_caching_settings.get("skip_existing", False))
                with gr.Row():
                    vae_path_wan = gr.Textbox(label="Wan2.1 VAE 文件路径", placeholder="例如：K:/models/wan2.1/vae.safetensors", value=wan_pre_caching_settings.get("vae_path", ""))
                    t5_path = gr.Textbox(label="T5 文件路径", placeholder="例如：K:/models/wan2.1/t5.pth", value=wan_pre_caching_settings.get("t5_path", ""))
                with gr.Row():
                    use_clip_checkbox_wan = gr.Checkbox(label="使用 CLIP 模型 (--clip)", value=wan_pre_caching_settings.get("use_clip", False))
                    clip_model_path_wan = gr.Textbox(label="Wan2.1 CLIP 模型文件路径", placeholder="例如：K:/models/wan2.1/clip.pth", visible=False, value=wan_pre_caching_settings.get("clip_model_path", ""))
                def toggle_clip_wan(checked):
                    return gr.update(visible=checked)
                use_clip_checkbox_wan.change(toggle_clip_wan, inputs=use_clip_checkbox_wan, outputs=clip_model_path_wan)
                with gr.Row():
                    run_cache_button_wan = gr.Button("运行 Wan2.1 预缓存")
                    stop_cache_button_wan = gr.Button("停止 Wan2.1 预缓存")
                cache_output_wan = gr.Textbox(label="Wan2.1 预缓存输出", lines=20, interactive=False)
                run_cache_button_wan.click(
                    fn=run_wan_cache_commands,
                    inputs=[dataset_config_file_wan, dataset_config_text_wan, enable_low_memory_wan, skip_existing_wan,
                            use_clip_checkbox_wan, clip_model_path_wan, vae_path_wan, t5_path],
                    outputs=cache_output_wan
                )
                stop_cache_button_wan.click(fn=stop_caching, inputs=None, outputs=cache_output_wan)

    ########################################
    # (2) Hunyuan Training / 训练 页面
    ########################################
    with gr.Tab("Training / 训练"):
        gr.Markdown("## Hunyuan 训练网络（toml文件上传，其它文件路径手动输入）")
        with gr.Row():
            dataset_config_file_train = gr.File(label="上传 dataset_config (toml)", file_count="single", file_types=[".toml"], type="filepath")
            dataset_config_text_train = gr.Textbox(label="或手动输入 toml 路径", placeholder="例如：K:/ai_software/config.toml", value=training_settings.get("dataset_config_text", ""))
        with gr.Row():
            max_train_epochs = gr.Number(label="训练 Epoch 数量 (>=2)", value=training_settings.get("max_train_epochs", 16), precision=0)
            learning_rate = gr.Textbox(label="学习率 (如 1e-4)", value=training_settings.get("learning_rate", "1e-4"))
        with gr.Row():
            network_dim = gr.Number(label="训练的 Dim (2-128)", value=training_settings.get("network_dim", 32), precision=0)
            network_alpha = gr.Number(label="训练的 Alpha (1-128)", value=training_settings.get("network_alpha", 16), precision=0)
        with gr.Row():
            gradient_accumulation_steps = gr.Number(label="梯度累积步数 (建议双数)", value=training_settings.get("gradient_accumulation_steps", 1), precision=0)
            enable_low_vram = gr.Checkbox(label="启用低 VRAM 模式", value=training_settings.get("enable_low_vram", False))
        blocks_to_swap = gr.Number(label="Blocks to Swap (20-36, 双数)", value=training_settings.get("blocks_to_swap", 20), precision=0, visible=training_settings.get("enable_low_vram", False))
        def toggle_blocks_swap(checked):
            return gr.update(visible=checked)
        enable_low_vram.change(toggle_blocks_swap, inputs=enable_low_vram, outputs=blocks_to_swap)
        with gr.Row():
            output_dir_input = gr.Textbox(label="输出目录", placeholder="./output", value=training_settings.get("output_dir", "./output"))
            output_name_input = gr.Textbox(label="输出名称 (例如 rem_test)", placeholder="rem_test", value=training_settings.get("output_name", "lora"))
        with gr.Row():
            save_every_n_epochs = gr.Number(label="每多少个 epoch 保存一次", value=training_settings.get("save_every_n_epochs", 1), precision=0)
        with gr.Row():
            use_network_weights = gr.Checkbox(label="从已有权重继续训练 (--network_weights)", value=training_settings.get("use_network_weights", False))
            network_weights_path = gr.Textbox(label="权重文件路径", placeholder="请输入权重文件路径", value=training_settings.get("network_weights_path", ""), visible=training_settings.get("use_network_weights", False))
        def toggle_network_weights_input(checked):
            return gr.update(visible=checked)
        use_network_weights.change(toggle_network_weights_input, inputs=use_network_weights, outputs=network_weights_path)
        with gr.Row():
            dit_weights_path = gr.Textbox(label="Hunyuan Dit 权重文件路径", placeholder="例如：K:/models/hunyuan/dit.pth")
        with gr.Row():
            sample_vae_path = gr.Textbox(label="Hunyuan Sample VAE 文件路径", placeholder="例如：K:/models/hunyuan/vae_sample.pth")
            sample_text_encoder1_path = gr.Textbox(label="Sample Text Encoder 1 路径", placeholder="例如：K:/models/hunyuan/text_encoder1_sample.pth")
            sample_text_encoder2_path = gr.Textbox(label="Sample Text Encoder 2 路径", placeholder="例如：K:/models/hunyuan/text_encoder2_sample.pth")
        with gr.Row():
            use_clip_checkbox_train = gr.Checkbox(label="使用 CLIP 模型 (--clip) （用于 I2V）", value=training_settings.get("use_clip", False))
            clip_model_path_train = gr.Textbox(label="Hunyuan CLIP 模型路径", placeholder="例如：K:/models/hunyuan/clip.pth", visible=False, value=training_settings.get("clip_model_path", ""))
        def toggle_clip_train(checked):
            return gr.update(visible=checked)
        use_clip_checkbox_train.change(toggle_clip_train, inputs=use_clip_checkbox_train, outputs=clip_model_path_train)
        with gr.Row():
            generate_samples_checkbox = gr.Checkbox(label="训练时生成范例 (sample_prompts)?", value=training_settings.get("generate_samples", False))
        with gr.Row():
            sample_every_n_epochs_input = gr.Number(label="sample_every_n_epochs", value=training_settings.get("sample_every_n_epochs", 1), precision=0, visible=training_settings.get("generate_samples", False))
            sample_every_n_steps_input = gr.Number(label="sample_every_n_steps", value=training_settings.get("sample_every_n_steps", 1000), precision=0, visible=training_settings.get("generate_samples", False))
        sample_prompt_text_input = gr.Textbox(label="Prompt 文本", value=training_settings.get("sample_prompt_text", "A cat walks on the grass, realistic style."), visible=training_settings.get("generate_samples", False))
        with gr.Row():
            sample_w_input = gr.Number(label="宽度 (w)", value=training_settings.get("sample_w", 640), precision=0, visible=training_settings.get("generate_samples", False))
            sample_h_input = gr.Number(label="高度 (h)", value=training_settings.get("sample_h", 480), precision=0, visible=training_settings.get("generate_samples", False))
            sample_frames_input = gr.Number(label="帧数 (f)", value=training_settings.get("sample_frames", 25), precision=0, visible=training_settings.get("generate_samples", False))
        with gr.Row():
            sample_seed_input = gr.Number(label="种子 (d)", value=training_settings.get("sample_seed", 123), precision=0, visible=training_settings.get("generate_samples", False))
            sample_steps_input = gr.Number(label="步数 (s)", value=training_settings.get("sample_steps", 20), precision=0, visible=training_settings.get("generate_samples", False))
        custom_prompt_txt_checkbox = gr.Checkbox(label="使用自定义 prompt_file (txt)？", value=training_settings.get("custom_prompt_txt", False), visible=training_settings.get("generate_samples", False))
        custom_prompt_path_input = gr.Textbox(label="自定义 prompt_file 路径", placeholder="请输入 prompt 文件路径", value=training_settings.get("custom_prompt_path", ""), visible=training_settings.get("generate_samples", False) and training_settings.get("custom_prompt_txt", False))
        # 增加上传 prompt_file.txt 的控件
        prompt_file_upload = gr.File(label="上传 prompt_file.txt（可选）", file_count="single", file_types=[".txt"], type="filepath")
        with gr.Row():
            run_train_button = gr.Button("运行训练")
            stop_train_button = gr.Button("停止训练")
        train_output = gr.Textbox(label="训练输出", lines=20, interactive=False)
        run_train_button.click(
            fn=run_training,
            inputs=[
                dataset_config_file_train, dataset_config_text_train,
                max_train_epochs, learning_rate,
                network_dim, network_alpha,
                gradient_accumulation_steps, enable_low_vram, blocks_to_swap,
                output_dir_input, output_name_input, save_every_n_epochs,
                use_network_weights, network_weights_path,
                use_clip_checkbox_train, clip_model_path_train,
                dit_weights_path,
                generate_samples_checkbox, sample_every_n_epochs_input, sample_every_n_steps_input,
                sample_prompt_text_input, sample_w_input, sample_h_input,
                sample_frames_input, sample_seed_input, sample_steps_input,
                custom_prompt_txt_checkbox, custom_prompt_path_input,
                prompt_file_upload,
                sample_vae_path, sample_text_encoder1_path, sample_text_encoder2_path
            ],
            outputs=train_output
        )
        stop_train_button.click(fn=stop_training, inputs=None, outputs=train_output)

    ########################################
    # (3) Wan2.1 Training / 训练 Wan2.1 页面
    ########################################
    with gr.Tab("Training Wan2.1 / 训练 Wan2.1"):
        gr.Markdown("## Wan2.1 训练网络（toml文件上传，其它文件路径手动输入）")
        with gr.Row():
            dataset_config_file_wan = gr.File(label="上传 dataset_config (toml)", file_count="single", file_types=[".toml"], type="filepath")
            dataset_config_text_wan = gr.Textbox(label="或手动输入 toml 路径", placeholder="例如：K:/ai_software/config.toml", value=wan_training_settings.get("dataset_config_text", ""))
        with gr.Row():
            task_dropdown = gr.Dropdown(label="任务 (task)", choices=["t2v-1.3B", "t2v-14B", "i2v-14B", "t2i-14B"], value=wan_training_settings.get("task", "t2v-1.3B"))
            dit_weights_path_wan = gr.Textbox(label="Dit 权重文件路径 (--dit)", placeholder="例如：K:/models/wan2.1/dit.safetensors", value=wan_training_settings.get("dit_weights_path", ""))
        with gr.Row():
            max_train_epochs_wan = gr.Number(label="训练 Epoch 数量 (>=2)", value=wan_training_settings.get("max_train_epochs", 16), precision=0)
            learning_rate_wan = gr.Textbox(label="学习率 (如 2e-4)", value=wan_training_settings.get("learning_rate", "2e-4"))
        with gr.Row():
            network_dim_wan = gr.Number(label="训练的 Dim (2-128)", value=wan_training_settings.get("network_dim", 32), precision=0)
            timestep_sampling_input = gr.Textbox(label="timestep_sampling", value=wan_training_settings.get("timestep_sampling", "shift"))
            discrete_flow_shift_input = gr.Number(label="discrete_flow_shift", value=wan_training_settings.get("discrete_flow_shift", 3.0), precision=1)
        with gr.Row():
            enable_low_vram_wan = gr.Checkbox(label="启用低 VRAM 模式", value=wan_training_settings.get("enable_low_vram", False))
            blocks_to_swap_wan = gr.Number(label="Blocks to Swap (20-36, 双数)", value=wan_training_settings.get("blocks_to_swap", 20), precision=0, visible=wan_training_settings.get("enable_low_vram", False))
        def toggle_blocks_swap_wan(checked):
            return gr.update(visible=checked)
        enable_low_vram_wan.change(toggle_blocks_swap_wan, inputs=enable_low_vram_wan, outputs=blocks_to_swap_wan)
        with gr.Row():
            output_dir_wan = gr.Textbox(label="输出目录", placeholder="./output", value=wan_training_settings.get("output_dir", "./output"))
            output_name_wan = gr.Textbox(label="输出名称", placeholder="wan_lora", value=wan_training_settings.get("output_name", "wan_lora"))
        with gr.Row():
            save_every_n_epochs_wan = gr.Number(label="每多少个 epoch 保存一次", value=wan_training_settings.get("save_every_n_epochs", 1), precision=0)
        with gr.Row():
            use_network_weights_wan = gr.Checkbox(label="从已有权重继续训练 (--network_weights)", value=wan_training_settings.get("use_network_weights", False))
            network_weights_path_wan = gr.Textbox(label="权重文件路径", placeholder="请输入权重文件路径", value=wan_training_settings.get("network_weights_path", ""), visible=wan_training_settings.get("use_network_weights", False))
        def toggle_network_weights_input_wan(checked):
            return gr.update(visible=checked)
        use_network_weights_wan.change(toggle_network_weights_input_wan, inputs=use_network_weights_wan, outputs=network_weights_path_wan)
        with gr.Row():
            use_clip_wan = gr.Checkbox(label="使用 CLIP 模型 (--clip) （用于 I2V）", value=wan_training_settings.get("use_clip", False))
            clip_model_path_wan = gr.Textbox(label="CLIP 模型路径", placeholder="例如：K:/models/wan2.1/clip.pth", value=wan_training_settings.get("clip_model_path", ""), visible=wan_training_settings.get("use_clip", False))
        def toggle_clip_input(checked):
            return gr.update(visible=checked)
        use_clip_wan.change(toggle_clip_input, inputs=use_clip_wan, outputs=clip_model_path_wan)
        with gr.Row():
            generate_samples_checkbox_wan = gr.Checkbox(label="训练时生成范例 (sample_prompts)?", value=wan_training_settings.get("generate_samples", False))
        with gr.Row():
            sample_every_n_epochs_wan = gr.Number(label="sample_every_n_epochs", value=wan_training_settings.get("sample_every_n_epochs", 1), precision=0, visible=wan_training_settings.get("generate_samples", False))
            sample_every_n_steps_wan = gr.Number(label="sample_every_n_steps", value=wan_training_settings.get("sample_every_n_steps", 1000), precision=0, visible=wan_training_settings.get("generate_samples", False))
        sample_prompt_text_wan = gr.Textbox(label="Prompt 文本", value=wan_training_settings.get("sample_prompt_text", "A beautiful landscape in Wan2.1 style."), visible=wan_training_settings.get("generate_samples", False))
        with gr.Row():
            sample_w_wan = gr.Number(label="宽度 (w)", value=wan_training_settings.get("sample_w", 832), precision=0, visible=wan_training_settings.get("generate_samples", False))
            sample_h_wan = gr.Number(label="高度 (h)", value=wan_training_settings.get("sample_h", 480), precision=0, visible=wan_training_settings.get("generate_samples", False))
            sample_frames_wan = gr.Number(label="帧数 (f)", value=wan_training_settings.get("sample_frames", 81), precision=0, visible=wan_training_settings.get("generate_samples", False))
        with gr.Row():
            sample_seed_wan = gr.Number(label="种子 (d)", value=wan_training_settings.get("sample_seed", 42), precision=0, visible=wan_training_settings.get("generate_samples", False))
            sample_steps_wan = gr.Number(label="步数 (s)", value=wan_training_settings.get("sample_steps", 20), precision=0, visible=wan_training_settings.get("generate_samples", False))
        custom_prompt_txt_checkbox_wan = gr.Checkbox(label="使用自定义 prompt_file (txt)？", value=wan_training_settings.get("custom_prompt_txt", False), visible=wan_training_settings.get("generate_samples", False))
        custom_prompt_path_wan = gr.Textbox(label="自定义 prompt_file 路径", placeholder="请输入 prompt 文件路径", value=wan_training_settings.get("custom_prompt_path", ""), visible=wan_training_settings.get("generate_samples", False) and wan_training_settings.get("custom_prompt_txt", False))
        # 增加上传 prompt_file.txt 的控件
        prompt_file_upload_wan = gr.File(label="上传 prompt_file.txt（可选）", file_count="single", file_types=[".txt"], type="filepath")
        with gr.Row():
            run_wan_train_button = gr.Button("运行 Wan2.1 训练")
            stop_wan_train_button = gr.Button("停止 Wan2.1 训练")
        wan_train_output = gr.Textbox(label="Wan2.1 训练输出", lines=20, interactive=False)
        run_wan_train_button.click(
            fn=run_wan_training,
            inputs=[
                dataset_config_file_wan, dataset_config_text_wan,
                task_dropdown, dit_weights_path_wan,
                max_train_epochs_wan, learning_rate_wan, network_dim_wan,
                enable_low_vram_wan, blocks_to_swap_wan,
                output_dir_wan, output_name_wan, save_every_n_epochs_wan,
                use_network_weights_wan, network_weights_path_wan,
                use_clip_wan, clip_model_path_wan,
                timestep_sampling_input, discrete_flow_shift_input,
                generate_samples_checkbox_wan, sample_every_n_epochs_wan, sample_every_n_steps_wan,
                sample_prompt_text_wan, sample_w_wan, sample_h_wan,
                sample_frames_wan, sample_seed_wan, sample_steps_wan,
                custom_prompt_txt_checkbox_wan, custom_prompt_path_wan,
                prompt_file_upload_wan,
                # Wan2.1 采样时使用 VAE 与 T5 文件路径，由用户输入
                gr.Textbox(label="VAE 文件路径 (--vae)", placeholder="例如：K:/models/wan2.1/vae.safetensors", value=wan_training_settings.get("sample_vae_path", "路径/to/wan_2.1_vae.safetensors")),
                gr.Textbox(label="T5 模型路径 (--t5)", placeholder="例如：K:/models/wan2.1/t5.pth", value=wan_training_settings.get("sample_t5_path", "路径/to/models_t5_umt5-xxl-enc-bf16.pth"))
            ],
            outputs=wan_train_output
        )
        stop_wan_train_button.click(fn=stop_training, inputs=None, outputs=wan_train_output)

    ########################################
    # (4) LoRA Conversion / LoRA 转换 页面
    ########################################
    with gr.Tab("LoRA Conversion / LoRA 转换"):
        gr.Markdown("## 将 LoRA 转换为其他格式 (target=other)")
        lora_file_path = gr.File(label="选择 Musubi LoRA 文件 (.safetensors)，仅获取路径", file_count="single", file_types=[".safetensors"], type="filepath")
        output_dir_conversion = gr.Textbox(label="输出目录 (可选)", placeholder="例如：K:/converted_output", value="")
        convert_button = gr.Button("Convert LoRA / 转换 LoRA")
        conversion_output = gr.Textbox(label="转换输出日志", lines=15, interactive=False)
        convert_button.click(fn=run_lora_conversion, inputs=[lora_file_path, output_dir_conversion], outputs=conversion_output)

    ########################################
    # 注意事项
    ########################################
    gr.Markdown("""
### 注意事项 / Notes 
1. **路径格式**：请使用正确的路径格式（Windows 下可使用正斜杠 `/` 或转义反斜杠 `\\`）。
2. **依赖项**：请确保嵌入式 Python 环境已安装 `accelerate`、`torch`、`numpy`、`psutil`、`gradio`、`toml` 等必要库。
3. **accelerate 配置**：首次使用时，请先运行 `./python_embeded/python.exe -m accelerate config` 进行配置。
4. **权限问题**：某些系统可能需要管理员权限运行命令。
5. **文件路径输入**：本应用中除 toml 文件外，其它模型文件均需用户手动输入文件路径（不会上传），请确保路径正确。
6. **Prompt File 上传**：训练时可上传 prompt_file.txt（可选），若上传则使用该文件进行推理。
7. **LoRA Conversion**：选择 `.safetensors` 文件后，输出文件名将自动添加 `_converted.safetensors` 后缀。
8. **训练续训**：在训练页面中，启用“从已有权重继续训练”后，请确保权重文件路径正确。
    """)

demo.queue()
demo.launch()
