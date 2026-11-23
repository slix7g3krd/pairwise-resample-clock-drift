# pairwise-resample-clock-drift

## 领域

游戏、图形与多媒体

## 主题

音频重采样时钟漂移

## 说明

设备实际时钟、欠载拉伸与重采样相位保持。

初始状态下 `media/resample_clock_drift.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
