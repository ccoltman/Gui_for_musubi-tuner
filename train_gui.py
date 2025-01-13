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
        # 先终止所有子进程
        for child in parent.children(recursive=True):
            child.terminate()
        # 最后终止父进程
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"[WARN] terminate_process_tree 出现异常: {e}")

def stop_caching():
    """
    停止当前正在运行的预缓存子进程 (cache_latents + cache_text_encoder_outputs).
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
    停止当前正在运行的训练子进程.
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
    """
    加载 settings.toml 文件中的设置。如果文件不存在，返回空字典。
    """
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
    """
    将设置保存到 settings.toml 文件中。
    """
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            toml.dump(settings, f)
    except Exception as e:
        print(f"[WARN] 保存 settings.toml 失败: {e}")

#########################
# 3. 处理输入数据集配置路径 (支持文本或文件)
#########################

def get_dataset_config(file_path: str, text_path: str) -> str:
    """
    根据用户输入的 file_path (str, 不上传) 和 text_path (str)，
    返回最终要使用的 toml 路径：
    - 若 file_path 不为空，优先使用 file_path
    - 否则使用 text_path
    - 如果都为空，返回空字符串
    """
    if file_path and os.path.isfile(file_path):
        return file_path
    elif text_path.strip():
        return text_path.strip()
    else:
        return ""

#########################
# 4. Pre-caching
#########################

def run_cache_commands(
    dataset_config_file: str,  # 仅获取路径
    dataset_config_text: str,
    enable_low_memory: bool,
    skip_existing: bool
) -> Generator[str, None, None]:
    """
    使用 generator 函数 + accumulated 文本，将所有输出追加到同一个文本框中。
    在控制台同样实时打印每行。
    """
    # 确定最终的 dataset_config 路径
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)

    python_executable = "./python_embeded/python.exe"

    # 第一段命令
    cache_latents_cmd = [
        python_executable, "cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", "./models/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
        "--vae_chunk_size", "32",
        "--vae_tiling"
    ]
    if enable_low_memory:
        cache_latents_cmd.extend(["--vae_spatial_tile_sample_min_size", "128", "--batch_size", "1"])
    if skip_existing:
        cache_latents_cmd.append("--skip_existing")

    # 第二段命令
    cache_text_encoder_cmd = [
        python_executable, "cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--text_encoder1", "./models/ckpts/text_encoder",
        "--text_encoder2", "./models/ckpts/text_encoder_2",
        "--batch_size", "16"
    ]
    if enable_low_memory:
        cache_text_encoder_cmd.append("--fp8_llm")

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["cache"] = process

        for line in process.stdout:
            print(line, end="", flush=True)  # 控制台实时输出
            accumulated += line
            yield accumulated

        return_code = process.wait()
        running_processes["cache"] = None

        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    # 运行第一段命令
    accumulated_main = "\n[INFO] 开始运行第一段预缓存 (cache_latents.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_latents_cmd):
        yield content
    accumulated_main += "\n[INFO] 第一段预缓存已完成。\n"
    yield accumulated_main

    # 运行第二段命令
    accumulated_main += "\n[INFO] 开始运行第二段预缓存 (cache_text_encoder_outputs.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_text_encoder_cmd):
        yield content
    accumulated_main += "\n[INFO] 第二段预缓存已完成。\n"
    yield accumulated_main

    # 保存预缓存设置到 settings.toml
    pre_caching_settings = {
        "pre_caching": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "enable_low_memory": enable_low_memory,
            "skip_existing": skip_existing
        }
    }
    # 读取现有设置，避免覆盖其他部分
    existing_settings = load_settings()
    existing_settings.update(pre_caching_settings)
    save_settings(existing_settings)

#########################
# 5. 训练函数 + 生成范例扩展
#########################

def make_prompt_file(
    prompt_text: str,
    w: int,
    h: int,
    frames: int,
    seed: int,
    steps: int,
    custom_prompt_txt: bool,
    custom_prompt_path: str
) -> str:
    """
    根据用户输入，生成 ./prompt_file.txt 或使用用户自定义的 prompt 文件。
    返回最终的 prompt 文件路径。
    """
    if custom_prompt_txt and custom_prompt_path.strip():
        # 用户自定义 prompt 文件
        return custom_prompt_path.strip()
    else:
        # 自动生成 prompt_file.txt
        default_prompt_path = "./prompt_file.txt"
        with open(default_prompt_path, "w", encoding="utf-8") as f:
            # 写一个例子
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
    # 新增参数：是否生成范例
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
    custom_prompt_path: str
) -> Generator[str, None, None]:
    """
    训练回调函数，新增“生成范例”逻辑
    """
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)

    python_executable = "./python_embeded/python.exe"

    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        "hv_train_network.py",
        "--dit", "models/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
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

    # 如果勾选了“生成范例”，添加对应参数
    if generate_samples:
        # 生成 prompt 文件
        prompt_file_path = make_prompt_file(
            prompt_text=sample_prompt_text,
            w=sample_w,
            h=sample_h,
            frames=sample_frames,
            seed=sample_seed,
            steps=sample_steps,
            custom_prompt_txt=custom_prompt_txt,
            custom_prompt_path=custom_prompt_path
        )
        # 添加 sample 参数
        command.extend([
            "--sample_prompts", prompt_file_path,
            "--sample_every_n_epochs", str(sample_every_n_epochs),
            "--sample_every_n_steps", str(sample_every_n_steps),
            "--sample_at_first",
            "--vae",  "./models/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
            "--vae_chunk_size",  "32",
            "--vae_spatial_tile_sample_min_size",  "128",
            "--text_encoder1",  "./models/ckpts/text_encoder",
            "--text_encoder2",  "./models/ckpts/text_encoder_2",
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
            # 生成范例相关设置
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
            "custom_prompt_path": custom_prompt_path
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
    accumulated_main = start_message

    for content in run_and_stream_output(command):
        yield content
        accumulated_main = content

    accumulated_main += "\n[INFO] 训练命令执行完成。\n"
    yield accumulated_main

#########################
# 6. LoRA Conversion (第三页)
#########################

def run_lora_conversion(lora_file_path: str, output_dir: str) -> Generator[str, None, None]:
    """
    - 用户只选择路径, 不上传文件
    - 命令: python convert_lora.py --input <in> --output <out> --target other
    - 输出名为 原文件名 + "_converted.safetensors"
    """
    if not lora_file_path or not os.path.isfile(lora_file_path):
        yield "[ERROR] 未选择有效的 LoRA 文件路径\n"
        return

    python_executable = "./python_embeded/python.exe"

    # 从 lora_file_path 中获取文件名
    in_path = lora_file_path  # 本地路径
    basename = os.path.basename(in_path)  # e.g. rem_lora.safetensors
    filename_no_ext, ext = os.path.splitext(basename)
    # 构造新文件名
    out_name = f"{filename_no_ext}_converted{ext}"  # e.g. rem_lora_converted.safetensors

    # 若 output_dir 为空，默认当前目录
    if not output_dir.strip():
        output_dir = "."

    # 拼接完整输出路径
    out_path = os.path.join(output_dir, out_name)

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
# 7. 构建 Gradio UI
#########################

# 加载上次的设置
settings = load_settings()

# 提取 pre_caching 和 training 相关设置
pre_caching_settings = settings.get("pre_caching", {})
training_settings = settings.get("training", {})

with gr.Blocks() as demo:
    gr.Markdown("# AI Software Musubi Tuner Gui code by Kohya Gui by TTP - 双语支持 (中/英)")

    ########################################
    # (1) Pre-caching 页面
    ########################################
    with gr.Tab("Pre-caching / 预缓存"):
        gr.Markdown("## Latent 和 Text Encoder 输出预缓存（仅选择路径，不上传文件） / Latent and Text Encoder Pre-caching (Path Selection Only)")

        with gr.Row():
            dataset_config_file_cache = gr.File(
                label="浏览选择 dataset_config (toml, 仅获取路径) / Browse dataset_config (toml, Path Only)",
                file_count="single",
                file_types=[".toml"],
                type="filepath",  # 仅返回本地路径
            )
            dataset_config_text_cache = gr.Textbox(
                label="或手动输入 toml 路径 / Or Enter toml Path Manually",
                value=pre_caching_settings.get("dataset_config_text", ""),
                placeholder="K:/ai_software/musubi-tuner/train/test/test.toml"
            )

        enable_low_memory = gr.Checkbox(
            label="启用低显存模式 (增加相关参数) / Enable Low VRAM Mode (Additional Parameters)",
            value=pre_caching_settings.get("enable_low_memory", False)
        )
        skip_existing = gr.Checkbox(
            label="是否跳过已存在的 Cache 文件 (--skip_existing) / Skip Existing Cache Files (--skip_existing)",
            value=pre_caching_settings.get("skip_existing", False)
        )

        with gr.Row():
            run_cache_button = gr.Button("运行预缓存 / Run Pre-caching")
            stop_cache_button = gr.Button("停止预缓存 / Stop Pre-caching")

        cache_output = gr.Textbox(
            label="预缓存输出（累加显示） / Pre-caching Output (Accumulated)",
            lines=20,
            interactive=False
        )

        run_cache_button.click(
            fn=run_cache_commands,
            inputs=[dataset_config_file_cache, dataset_config_text_cache, enable_low_memory, skip_existing],
            outputs=cache_output
        )

        stop_cache_button.click(
            fn=stop_caching,
            inputs=None,
            outputs=cache_output
        )

    ########################################
    # (2) Training 页面
    ########################################
    with gr.Tab("Training / 训练"):
        gr.Markdown("## HV 训练网络（仅选择路径，不上传文件） / HV Training Network (Path Selection Only)")

        with gr.Row():
            dataset_config_file_train = gr.File(
                label="浏览选择 dataset_config (toml, 仅获取路径) / Browse dataset_config (toml, Path Only)",
                file_count="single",
                file_types=[".toml"],
                type="filepath",  # 仅返回本地路径
            )
            dataset_config_text_train = gr.Textbox(
                label="或手动输入 toml 路径 / Or Enter toml Path Manually",
                value=training_settings.get("dataset_config_text", ""),
                placeholder="K:/ai_software/musubi-tuner/train/test/test.toml"
            )

        with gr.Row():
            max_train_epochs = gr.Number(
                label="训练 Epoch 数量 (>=2) / Number of Training Epochs (>=2)",
                value=training_settings.get("max_train_epochs", 16),
                precision=0
            )
            learning_rate = gr.Textbox(
                label="学习率 (如 1e-4) / Learning Rate (e.g., 1e-4)",
                value=training_settings.get("learning_rate", "1e-4")
            )

        with gr.Row():
            network_dim = gr.Number(
                label="训练的Dim (2-128) / Training Dim (2-128)",
                value=training_settings.get("network_dim", 32),
                precision=0
            )
            network_alpha = gr.Number(
                label="训练的Alpha (1-128) / Training Alpha (1-128)",
                value=training_settings.get("network_alpha", 16),
                precision=0
            )

        with gr.Row():
            gradient_accumulation_steps = gr.Number(
                label="梯度累积步数 (建议双数) / Gradient Accumulation Steps (Even Number Recommended)",
                value=training_settings.get("gradient_accumulation_steps", 1),
                precision=0
            )
            enable_low_vram = gr.Checkbox(
                label="启用低 VRAM 模式 / Enable Low VRAM Mode",
                value=training_settings.get("enable_low_vram", False)
            )

        blocks_to_swap = gr.Number(
            label="Blocks to Swap (20-36, 双数) / Blocks to Swap (20-36, Even Number)",
            value=training_settings.get("blocks_to_swap", 20),
            precision=0,
            visible=training_settings.get("enable_low_vram", False)
        )

        def toggle_blocks_swap(checked):
            return gr.update(visible=checked)

        enable_low_vram.change(
            toggle_blocks_swap,
            inputs=enable_low_vram,
            outputs=blocks_to_swap
        )

        with gr.Row():
            output_dir_input = gr.Textbox(
                label="Output Directory / 输出目录",
                value=training_settings.get("output_dir", "./output"),
                placeholder="./output"
            )
            output_name_input = gr.Textbox(
                label="Output Name (e.g., rem_test) / 输出名称 (例如 rem_test)",
                value=training_settings.get("output_name", "lora"),
                placeholder="rem_test"
            )

        # 新增行：save_every_n_epochs
        with gr.Row():
            save_every_n_epochs = gr.Number(
                label="每多少个 epoch 保存一次 (save_every_n_epochs) / Save Every N Epochs (save_every_n_epochs)",
                value=training_settings.get("save_every_n_epochs", 1),
                precision=0
            )

        # 新增行：使用已有权重
        with gr.Row():
            use_network_weights = gr.Checkbox(
                label="从已有权重继续训练 (--network_weights) / Continue Training from Existing Weights (--network_weights)",
                value=training_settings.get("use_network_weights", False)
            )
            network_weights_path = gr.Textbox(
                label="权重文件路径 / Weights File Path",
                placeholder="path/to/weights_file.safetensors",
                value=training_settings.get("network_weights_path", ""),
                visible=training_settings.get("use_network_weights", False)
            )

        # 根据复选框显示权重路径输入
        def toggle_network_weights_input(checked):
            return gr.update(visible=checked)

        use_network_weights.change(
            toggle_network_weights_input,
            inputs=use_network_weights,
            outputs=network_weights_path
        )

        # 新增：生成范例相关参数
        generate_samples_checkbox = gr.Checkbox(
            label="训练时生成范例 (sample_prompts)?",
            value=training_settings.get("generate_samples", False)
        )
        sample_every_n_epochs_input = gr.Number(
            label="sample_every_n_epochs",
            value=training_settings.get("sample_every_n_epochs", 1),
            precision=0,
            visible=training_settings.get("generate_samples", False)
        )
        sample_every_n_steps_input = gr.Number(
            label="sample_every_n_steps",
            value=training_settings.get("sample_every_n_steps", 1000),
            precision=0,
            visible=training_settings.get("generate_samples", False)
        )
        sample_prompt_text_input = gr.Textbox(
            label="Prompt 文本 / Prompt Text",
            value=training_settings.get("sample_prompt_text", "A cat walks on the grass, realistic style."),
            visible=training_settings.get("generate_samples", False)
        )
        sample_w_input = gr.Number(label="宽度 (w)", value=training_settings.get("sample_w", 640), precision=0, visible=training_settings.get("generate_samples", False))
        sample_h_input = gr.Number(label="高度 (h)", value=training_settings.get("sample_h", 480), precision=0, visible=training_settings.get("generate_samples", False))
        sample_frames_input = gr.Number(label="帧数 (f)", value=training_settings.get("sample_frames", 25), precision=0, visible=training_settings.get("generate_samples", False))
        sample_seed_input = gr.Number(label="种子 (d)", value=training_settings.get("sample_seed", 123), precision=0, visible=training_settings.get("generate_samples", False))
        sample_steps_input = gr.Number(label="步数 (s)", value=training_settings.get("sample_steps", 20), precision=0, visible=training_settings.get("generate_samples", False))

        custom_prompt_txt_checkbox = gr.Checkbox(
            label="使用自定义 prompt_file (txt)？",
            value=training_settings.get("custom_prompt_txt", False),
            visible=training_settings.get("generate_samples", False)
        )
        custom_prompt_path_input = gr.Textbox(
            label="自定义 prompt_file 路径",
            value=training_settings.get("custom_prompt_path", ""),
            placeholder="K:/my_prompt.txt",
            visible=training_settings.get("generate_samples", False) and training_settings.get("custom_prompt_txt", False)
        )

        # 当 generate_samples_checkbox 勾选时，显示 sample 参数
        def toggle_generate_samples(checked, checked2):
            updates = []
            # 使 sample_every_n_epochs_input ... sample_steps_input 全部可见
            for _ in range(8):
                updates.append(gr.update(visible=checked))
            # 同时使 custom_prompt_txt_checkbox 可见
            updates.append(gr.update(visible=checked))
            # custom_prompt_path_input 的可见性根据 checked + checked2
            # 这里先简单设置为 not generate_samples => not visible
            custom_prompt_visible = checked and checked2
            updates.append(gr.update(visible=custom_prompt_visible))
            return updates

        generate_samples_checkbox.change(
            toggle_generate_samples,
            inputs=[generate_samples_checkbox, custom_prompt_txt_checkbox],
            outputs=[
                sample_every_n_epochs_input,
                sample_every_n_steps_input,
                sample_prompt_text_input,
                sample_w_input,
                sample_h_input,
                sample_frames_input,
                sample_seed_input,
                sample_steps_input,
                custom_prompt_txt_checkbox,
                custom_prompt_path_input
            ]
        )

        # 当 custom_prompt_txt_checkbox 勾选时，也需要更新 prompt_path 的可见性
        def toggle_custom_prompt(checked, generate_checked):
            # 只有当 generate_samples_checkbox 和 custom_prompt_txt_checkbox 都勾选时，才显示
            visible = checked and generate_checked
            return gr.update(visible=visible)

        custom_prompt_txt_checkbox.change(
            toggle_custom_prompt,
            inputs=[custom_prompt_txt_checkbox, generate_samples_checkbox],
            outputs=custom_prompt_path_input
        )

        with gr.Row():
            run_train_button = gr.Button("运行训练 / Run Training")
            stop_train_button = gr.Button("停止训练 / Stop Training")

        train_output = gr.Textbox(
            label="训练输出 / Training Output",
            lines=20,
            interactive=False
        )

        run_train_button.click(
            fn=run_training,
            inputs=[
                dataset_config_file_train,
                dataset_config_text_train,
                max_train_epochs,
                learning_rate,
                network_dim,
                network_alpha,
                gradient_accumulation_steps,
                enable_low_vram,
                blocks_to_swap,
                output_dir_input,
                output_name_input,
                save_every_n_epochs,
                use_network_weights,
                network_weights_path,
                # === 新增的生成范例参数 ===
                generate_samples_checkbox,
                sample_every_n_epochs_input,
                sample_every_n_steps_input,
                sample_prompt_text_input,
                sample_w_input,
                sample_h_input,
                sample_frames_input,
                sample_seed_input,
                sample_steps_input,
                custom_prompt_txt_checkbox,
                custom_prompt_path_input
            ],
            outputs=train_output
        )

        stop_train_button.click(
            fn=stop_training,
            inputs=None,
            outputs=train_output
        )

    ########################################
    # (3) LoRA Conversion 页面
    ########################################
    with gr.Tab("LoRA Conversion / LoRA 转换"):
        gr.Markdown("## 将 LoRA 转换为其他格式 (target=other) / Convert LoRA to Other Formats (target=other)")

        lora_file_input = gr.File(
            label="选择 Musubi LoRA 文件 (.safetensors)，仅获取路径 / Select Musubi LoRA File (.safetensors), Path Only",
            file_count="single",
            file_types=[".safetensors"],
            type="filepath"  # 仅返回本地路径
        )
        output_dir_conversion = gr.Textbox(
            label="输出目录 (可选)，若不填则默认当前目录 / Output Directory (Optional, Defaults to Current Directory)",
            value="",
            placeholder="K:/ai_software/musubi-tuner/converted_output"
        )

        convert_button = gr.Button("Convert LoRA / 转换 LoRA")
        conversion_output = gr.Textbox(
            label="转换输出日志 / Conversion Output Logs",
            lines=15,
            interactive=False
        )

        convert_button.click(
            fn=run_lora_conversion,
            inputs=[lora_file_input, output_dir_conversion],
            outputs=conversion_output
        )

    ########################################
    # 注意事项
    ########################################
    gr.Markdown("""
### 注意事项 / Notes 
1. **路径格式 / Path Format**：请使用正确路径格式 (Windows可用正斜杠 /，或转义反斜杠 \\)。  
2. **依赖项 / Dependencies**：确认嵌入式 Python 环境已安装 `accelerate`, `torch`, `numpy`, `psutil`, `gradio`, `toml` 等必要库。  
3. **accelerate 配置 / Accelerate Configuration**：如首次使用，需要先运行 `./python_embeded/python.exe -m accelerate config` 配置。  
4. **权限问题 / Permission Issues**：在某些操作系统中，需要管理员权限执行命令。  
5. **LoRA Conversion**：输入 `.safetensors` 路径，不上传文件，输出会自动在文件名后加 `_converted.safetensors`。  
6. **训练续训功能 / Training Continuation**：在“Training”标签页中，启用“从已有权重继续训练”后，请确保输入的权重文件路径正确。
    """)

demo.queue()
demo.launch()
