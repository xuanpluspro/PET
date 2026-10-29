# 断点续训

`main.py` 与 `train_wheat_full.py` 均在每个完整 epoch（包括该轮计划的验证）结束后保存：

- `checkpoint.pth`：最新完整训练状态，用于恢复。
- `best_checkpoint.pth`：验证 MAE 最低的状态，用于评估。

状态包含模型、优化器、学习率调度器、已完成 epoch、最佳 MAE、Python/NumPy/PyTorch 随机状态；全图训练入口还保存 AMP GradScaler。多卡时分别保存各进程的随机状态。

## 常规训练与自动恢复

在仓库根目录运行，路径和参数按实际实验调整：

```bash
python main.py \
  --dataset_file Wheat1799 \
  --data_path /root/gpufree-data/PET/data/RiceW3999 \
  --batch_size 8 --epochs 50 \
  --output_dir ricew3999_ep50_bs8 \
  --auto-resume
```

首次运行没有检查点，会正常开始；中断后执行同一条命令，会自动恢复
`outputs/Wheat1799/ricew3999_ep50_bs8/checkpoint.pth`。
也可以在原训练命令后加：

```bash
--resume outputs/Wheat1799/ricew3999_ep50_bs8/checkpoint.pth
```

`--resume` 也接受实验目录，并优先于 `--auto-resume`。
全图入口用法相同：在原 `python train_wheat_full.py ...` 命令后加 `--auto-resume`，
检查点位于 `outputs/Wheat1799Full/<output_dir>/`。全图入口只适合满足其尺寸要求的数据。

## 恢复规则与限制

- `--epochs 50` 表示总共训练 50 轮。例如已完成 20 轮，恢复后运行第 21–50 轮。保持原实验的总轮数、架构、学习率和损失配置；不兼容的配置会明确报错。
- 爆显存或关机后，从上一个保存成功的完整 epoch 恢复，当前未保存的一轮会重跑；不是从某个 batch 精确恢复。第一个 epoch 尚未保存时无法恢复。
- 可以因显存不足调小 batch size；代码会提示训练轨迹发生变化。要尽量复现实验，应保持数据、batch size、进程数和运行环境一致。GPU 的非确定性仍可能造成差异。
- 写入使用同目录临时文件、刷新和原子替换；写入失败时保留原检查点。存储盘被清空或实例删除不在此保护范围内。
- 旧版完整检查点可继续训练，但缺少随机状态时会警告；只有模型权重的文件不能恢复优化器等训练状态。仅加载可信的训练检查点。
- 恢复时清理 CSV 中晚于检查点的指标行，避免因保存前中断造成重复记录；重新执行的逐轮评估会覆盖对应目录。文本日志仍保留历史运行记录。
- 同一输出目录已有检查点但未指定恢复选项时会报错，防止误覆盖。新实验请使用新的 `--output_dir`。

## 验证

```bash
python -m unittest discover -s tests -p test_checkpointing.py -v
```

测试覆盖连续训练/恢复训练一致性、随机状态、优化器与调度器、AMP 状态接口、旧版兼容、参数检查、原子写入失败保护及指标去重。多进程测试在禁止 Unix socket 的环境中自动跳过；CUDA/DDP 需要在实际 GPU 环境验证。
