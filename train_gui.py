import gradio as gr
import subprocess
import sys
import os
import signal
import psutil
from typing import Generator

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
# 2. 处理输入数据集配置路径 (支持文本或文件)
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
# 3. Pre-caching
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

#########################
# 4. Training
#########################

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
    save_every_n_epochs: int
) -> Generator[str, None, None]:
    """
    训练命令同样使用 accumulated 追加方式，并可被中止。
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
        accumulated_main = content  # 最后一次 content 包含全部日志

    accumulated_main += "\n[INFO] 训练命令执行完成。\n"
    yield accumulated_main

#########################
# 5. LoRA Conversion (第三页)
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
# 构建 Gradio UI
#########################
with gr.Blocks() as demo:
    gr.Markdown("# AI Software Musubi Tuner Gui - Code from Kohya, Gui by TTP")

    ########################################
    # (1) Pre-caching 页面
    ########################################
    with gr.Tab("Pre-caching"):
        gr.Markdown("## Latent 和 Text Encoder 输出预缓存")

        with gr.Row():
            dataset_config_file_cache = gr.File(
                label="浏览选择 dataset_config (toml)",
                file_count="single",
                file_types=[".toml"],
                type="filepath",  # 仅返回本地路径
            )
            dataset_config_text_cache = gr.Textbox(
                label="或手动输入 toml 路径",
                placeholder="K:/ai_software/musubi-tuner/train/test/test.toml"
            )

        enable_low_memory = gr.Checkbox(
            label="启用低显存模式",
            value=False
        )
        skip_existing = gr.Checkbox(
            label="是否跳过已存在的 Cache 文件 (--skip_existing)",
            value=False
        )

        with gr.Row():
            run_cache_button = gr.Button("运行预缓存")
            stop_cache_button = gr.Button("停止预缓存")

        cache_output = gr.Textbox(
            label="预缓存输出",
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
    with gr.Tab("Training"):
        gr.Markdown("## HV 训练网络")

        with gr.Row():
            dataset_config_file_train = gr.File(
                label="浏览选择 dataset_config (toml)",
                file_count="single",
                file_types=[".toml"],
                type="filepath",  # 仅返回本地路径
            )
            dataset_config_text_train = gr.Textbox(
                label="或手动输入 toml 路径",
                placeholder="K:/ai_software/musubi-tuner/train/test/test.toml"
            )

        with gr.Row():
            max_train_epochs = gr.Number(
                label="训练 Epoch 数量 (>=2)",
                value=16,
                precision=0
            )
            learning_rate = gr.Textbox(
                label="学习率 (如 1e-4)",
                value="1e-4"
            )

        with gr.Row():
            network_dim = gr.Number(
                label="训练的Dim (2-128)",
                value=32,
                precision=0
            )
            network_alpha = gr.Number(
                label="训练的Alpha (1-128)",
                value=16,
                precision=0
            )

        with gr.Row():
            gradient_accumulation_steps = gr.Number(
                label="梯度累积步数 (建议双数)",
                value=1,
                precision=0
            )
            enable_low_vram = gr.Checkbox(
                label="启用低 VRAM 模式",
                value=False
            )

        blocks_to_swap = gr.Number(
            label="Blocks to Swap (20-36, 双数)",
            value=20,
            precision=0,
            visible=False
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
                label="Output Directory",
                value="./output",
                placeholder="./output"
            )
            output_name_input = gr.Textbox(
                label="Output Name (e.g., rem_test)",
                value="lora",
                placeholder="rem_test"
            )

        with gr.Row():
            save_every_n_epochs = gr.Number(
                label="每多少个 epoch 保存一次 (save_every_n_epochs)",
                value=1,
                precision=0
            )

        with gr.Row():
            run_train_button = gr.Button("运行训练")
            stop_train_button = gr.Button("停止训练")

        train_output = gr.Textbox(
            label="训练输出",
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
                save_every_n_epochs
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
    with gr.Tab("LoRA Conversion"):
        gr.Markdown("## 将 LoRA 转换为其他格式(comfyui)兼容")

        lora_file_input = gr.File(
            label="选择 Musubi LoRA 文件 (.safetensors)，仅获取路径",
            file_count="single",
            file_types=[".safetensors"],
            type="filepath"  # 仅返回本地路径
        )
        output_dir_conversion = gr.Textbox(
            label="输出目录 (可选)，若不填则默认当前目录",
            value="./output",
            placeholder="K:/ai_software/musubi-tuner/converted_output"
        )

        convert_button = gr.Button("Convert LoRA")
        conversion_output = gr.Textbox(
            label="转换输出日志",
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
### 注意事项
1. **路径格式**：请使用正确路径格式 (Windows可用正斜杠 /，或转义反斜杠 \\)。  
2. **LoRA Conversion**：输入 `.safetensors` 路径，不上传文件，输出会自动在文件名后加 `_converted.safetensors`。
    """)

demo.queue()
demo.launch()
