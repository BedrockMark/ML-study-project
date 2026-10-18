import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="解析 HuggingFace trainer_state.json 并绘制训练指标曲线图"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="trainer_state.json 文件的路径 (支持相对路径或绝对路径)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="图表保存的输出文件夹路径 (支持相对路径或绝对路径)",
    )
    return parser.parse_args()


def load_trainer_state(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"找不到指定的文件: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "log_history" not in data:
        raise KeyError(
            f"文件 {json_path} 中未找到 'log_history' 字段，请确认文件格式！"
        )

    return data["log_history"]


def extract_metrics(log_history):
    # 提取训练指标
    train_steps = []
    train_loss = []
    learning_rates = []
    lr_steps = []
    grad_norms = []
    grad_steps = []

    # 兼容可能存在的评估指标 (Eval)
    eval_steps = []
    eval_loss = []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue

        # 训练 Loss
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])

        # 学习率
        if "learning_rate" in entry:
            lr_steps.append(step)
            learning_rates.append(entry["learning_rate"])

        # 梯度范数
        if "grad_norm" in entry:
            grad_steps.append(step)
            grad_norms.append(entry["grad_norm"])

        # 验证集 Loss（若有）
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])

    return {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "lr_steps": lr_steps,
        "learning_rates": learning_rates,
        "grad_steps": grad_steps,
        "grad_norms": grad_norms,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
    }


def plot_and_save(metrics, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 绘制综合 Dashboard (三合一面板)
    plt.figure(figsize=(15, 10))

    # 子图 1: Loss 曲线
    plt.subplot(2, 2, 1)
    if metrics["train_loss"]:
        plt.plot(
            metrics["train_steps"],
            metrics["train_loss"],
            label="Train Loss",
            color="#1f77b4",
            alpha=0.8,
        )
    if metrics["eval_loss"]:
        plt.plot(
            metrics["eval_steps"],
            metrics["eval_loss"],
            label="Eval Loss",
            color="#ff7f0e",
            linestyle="--",
            marker="o",
            markersize=3,
        )
    plt.title("Loss vs. Step", fontsize=12, fontweight="bold")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 子图 2: Learning Rate 曲线
    plt.subplot(2, 2, 2)
    if metrics["learning_rates"]:
        plt.plot(
            metrics["lr_steps"],
            metrics["learning_rates"],
            label="Learning Rate",
            color="#2ca02c",
        )
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.title("Learning Rate vs. Step", fontsize=12, fontweight="bold")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 子图 3: Gradient Norm 曲线
    plt.subplot(2, 2, 3)
    if metrics["grad_norms"]:
        plt.plot(
            metrics["grad_steps"],
            metrics["grad_norms"],
            label="Grad Norm",
            color="#d62728",
            alpha=0.6,
        )
    plt.title("Gradient Norm vs. Step", fontsize=12, fontweight="bold")
    plt.xlabel("Step")
    plt.ylabel("Grad Norm")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    summary_path = output_dir / "training_summary.png"
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"[✔] 综合指标图已保存至: {summary_path}")

    # 2. 单独保存 Loss 曲线图 (高清单独版)
    if metrics["train_loss"]:
        plt.figure(figsize=(8, 5))
        plt.plot(
            metrics["train_steps"],
            metrics["train_loss"],
            label="Train Loss",
            color="#1f77b4",
        )
        if metrics["eval_loss"]:
            plt.plot(
                metrics["eval_steps"],
                metrics["eval_loss"],
                label="Eval Loss",
                color="#ff7f0e",
                linestyle="--",
            )
        plt.title("Training & Eval Loss", fontsize=12)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        loss_path = output_dir / "loss_curve.png"
        plt.savefig(loss_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[✔] Loss 曲线图已保存至: {loss_path}")


def main():
    args = parse_args()

    # 处理相对路径转绝对路径
    input_file = Path(args.input_path).resolve()
    output_directory = Path(args.output_dir).resolve()

    print(f"[*] 正在读取: {input_file}")
    log_history = load_trainer_state(input_file)

    print(f"[*] 正在提取训练日志数据（共 {len(log_history)} 条记录）...")
    metrics = extract_metrics(log_history)

    print(f"[*] 正在生成可视化图表...")
    plot_and_save(metrics, output_directory)

    print("[*] 全部完成！")


if __name__ == "__main__":
    main()