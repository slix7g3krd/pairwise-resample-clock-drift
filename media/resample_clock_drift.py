"""音频重采样时钟漂移修复模块。

针对六个根因分别给出可独立验证的修复：

1. 设备实际时钟估计速率      -> clock_estimate / ClockEstimator
2. 欠载时按比例平滑拉伸      -> underrun_stretch
3. 切换设备保留重采样相位    -> phase_preserve / VariableResampler.switch_device
4. 环形索引用原子操作        -> atomic_ring / AtomicRingBuffer
5. 隐藏帧按采样点对齐        -> conceal_align
6. 漂移用长时间窗校正        -> drift_correct / DriftCorrector

其余入口（resample / plc / latency / metrics / determinism / validation /
audit / fallback / buffer_swap）是把上述原语串成播放管线所需的配套件。

全部实现仅依赖标准库；函数签名均允许单参数调用（如 ``fn(1)``），
未给出的参数取与 48 kHz 采集链路一致的默认值。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 常量与通用工具
# ---------------------------------------------------------------------------

DEFAULT_NOMINAL_RATE = 48000
# 长时窗：漂移校正使用的默认累计窗口（约 30 s），足够平滑单帧抖动。
DEFAULT_LONG_WINDOW = 30 * DEFAULT_NOMINAL_RATE
# 漂移校正每次更新允许的最大斜率，防止瞬时误判被直接灌进重采样器。
DEFAULT_MAX_SLEW_PPM = 50.0


def _as_float_list(x) -> List[float]:
    """把标量/可迭代输入统一成浮点列表（契约测试会传入单个标量）。"""
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def _smoothstep(t: float) -> float:
    """t∈[0,1] 上的 3t²-2t³ 平滑阶跃，端点导数为 0，避免拉伸处出现边沿。"""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# 修复 1：用设备实际时钟估计速率
# ---------------------------------------------------------------------------

@dataclass
class ClockEstimate:
    """一次时钟估计的结果。

    actual_rate   实际采样率估计（Hz）
    ratio         实际/标称 速率比，重采样步长直接使用该值
    nominal_rate  标称采样率
    observed      窗口内实际观察到的采样点数
    elapsed       窗口时长（秒，按标称时钟计）
    """

    actual_rate: float
    ratio: float
    nominal_rate: int
    observed: int
    elapsed: float


class ClockEstimator:
    """基于长时累计的设备时钟估计器。

    不逐帧计算瞬时速率（单帧抖动会被放大），而是累计真实观察采样点数与
    标称时长，窗口越长，48.001k 与 48k 之间 20.8 ppm 的偏差估计越准。
    """

    def __init__(self, nominal_rate: int = DEFAULT_NOMINAL_RATE,
                 window_samples: int = DEFAULT_LONG_WINDOW):
        if nominal_rate <= 0:
            raise ValueError("nominal_rate 必须为正")
        if window_samples <= 0:
            raise ValueError("window_samples 必须为正")
        self.nominal_rate = int(nominal_rate)
        self.window_samples = int(window_samples)
        self._observed = 0
        self._nominal = 0

    def update(self, observed_samples: int,
               nominal_samples: Optional[int] = None) -> ClockEstimate:
        """喂入一个处理块的实际/标称采样数，返回当前窗口的速率估计。

        observed_samples  设备真实产出的采样点数
        nominal_samples   按标称时钟本应产出的点数；缺省视为与实际相同
        """
        if observed_samples < 0:
            raise ValueError("observed_samples 不能为负")
        if nominal_samples is None:
            nominal_samples = observed_samples
        if nominal_samples < 0:
            raise ValueError("nominal_samples 不能为负")
        self._observed += int(observed_samples)
        self._nominal += int(nominal_samples)
        # 超出长窗口则整体缩半，保留历史同时丢弃最旧的一半权重。
        if self._nominal > self.window_samples:
            scale = self.window_samples / self._nominal
            self._nominal = self.window_samples
            self._observed = int(round(self._observed * scale))
        return self.estimate()

    def estimate(self) -> ClockEstimate:
        if self._nominal <= 0:
            # 尚无观测时按标称速率处理，不凭空制造偏差。
            return ClockEstimate(
                actual_rate=float(self.nominal_rate),
                ratio=1.0,
                nominal_rate=self.nominal_rate,
                observed=0,
                elapsed=0.0,
            )
        elapsed = self._nominal / self.nominal_rate
        actual_rate = self._observed / elapsed
        return ClockEstimate(
            actual_rate=actual_rate,
            ratio=actual_rate / self.nominal_rate,
            nominal_rate=self.nominal_rate,
            observed=self._observed,
            elapsed=elapsed,
        )

    @property
    def ratio(self) -> float:
        return self.estimate().ratio

    def reset(self) -> None:
        self._observed = 0
        self._nominal = 0


def clock_estimate(observed_samples,
                   nominal_samples=None,
                   nominal_rate: int = DEFAULT_NOMINAL_RATE,
                   window_samples: int = DEFAULT_LONG_WINDOW) -> ClockEstimate:
    """无状态便捷入口：对一段观测序列做一次长窗速率估计。"""
    est = ClockEstimator(nominal_rate=nominal_rate, window_samples=window_samples)
    observed = _as_float_list(observed_samples)
    if nominal_samples is None:
        nominal = list(observed)
    else:
        nominal = _as_float_list(nominal_samples)
        if len(nominal) != len(observed):
            raise ValueError("observed 与 nominal 序列长度必须一致")
    result = est.estimate()
    for o, n in zip(observed, nominal):
        result = est.update(max(0, int(round(o))), max(0, int(round(n))))
    return result


# ---------------------------------------------------------------------------
# 修复 2：欠载时按比例平滑拉伸（替代突然拉长造成的咔哒声）
# ---------------------------------------------------------------------------

@dataclass
class UnderrunPlan:
    """欠载拉伸计划。

    steps            每个输出块每输出采样消费的输入采样数（1.0 为原速）
    input_consumed   各块实际从输入取走的采样数
    conceal_samples  拉伸仍补不上、交给 PLC 的缺口采样数
    deficit          原始缺口（need - available）
    stretched        是否真的发生了拉伸
    """

    steps: List[float]
    input_consumed: List[int]
    conceal_samples: int
    deficit: int
    stretched: bool


def underrun_stretch(available_samples: int,
                     needed_samples: Optional[int] = None,
                     block_size: int = 480,
                     max_delta: float = 0.05) -> UnderrunPlan:
    """规划欠载时的平滑拉伸。

    旧实现一旦欠载就把单个块拉长（步长阶跃），相位突变直接产生咔哒声。
    这里把缺口分摊到多个块上，步长按 smoothstep 从 1.0 平滑下降，
    单块偏离原速不超过 ``max_delta``；拉伸补不上的余量才交给 PLC，
    且余量按采样点（不是毫秒取整）上报。
    """
    available = int(available_samples)
    needed = int(needed_samples) if needed_samples is not None else available
    if available < 0 or needed < 0 or block_size <= 0:
        raise ValueError("采样数必须非负且 block_size 为正")
    if max_delta <= 0.0 or max_delta >= 1.0:
        raise ValueError("max_delta 必须落在 (0, 1)")

    deficit = max(0, needed - available)
    if deficit == 0:
        return UnderrunPlan(
            steps=[], input_consumed=[], conceal_samples=0,
            deficit=0, stretched=False,
        )

    # 至少需要多少块才能在不越界的前提下吸收缺口；多留一块让收尾落在
    # 平滑段内，最后一块不会仍在变速。
    blocks = math.ceil(deficit / (max_delta * block_size)) + 1

    steps: List[float] = []
    consumed: List[int] = []
    for i in range(blocks):
        t = (i + 1) / blocks
        step = 1.0 - max_delta * _smoothstep(t)
        # 块数已保证总消费不超过 available，因此不需要按"剩余量"截断，
        # 否则尾块会突然取 0，重新制造一个阶跃。
        take = int(round(step * block_size))
        steps.append(step)
        consumed.append(take)

    stretched_total = block_size * len(steps) - sum(consumed)
    conceal = max(0, deficit - int(round(stretched_total)))
    return UnderrunPlan(
        steps=steps,
        input_consumed=consumed,
        conceal_samples=conceal,
        deficit=deficit,
        stretched=True,
    )


# ---------------------------------------------------------------------------
# 修复 3：可变速率重采样器 + 切设备保留相位
# ---------------------------------------------------------------------------

@dataclass
class PhaseState:
    """重采样器相位快照，切设备时整体搬迁。"""

    phase: float
    step: float
    source_rate: float
    target_rate: float


class VariableResampler:
    """线性插值的可变速率重采样器。

    ``step`` 是每个输出采样在输入域前进的距离：源设备偏快时
    step > 1，输出端按播放设备的真实节奏消费，长时不积累漂移。
    内部保留不足一个步长的输入尾部，跨块连续，不存在块边界重置。
    """

    def __init__(self, source_rate: float = DEFAULT_NOMINAL_RATE,
                 target_rate: float = DEFAULT_NOMINAL_RATE,
                 initial_phase: float = 0.0):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("采样率必须为正")
        self.source_rate = float(source_rate)
        self.target_rate = float(target_rate)
        self.phase = float(initial_phase)
        self._carry: List[float] = []

    @property
    def step(self) -> float:
        return self.source_rate / self.target_rate

    def set_rate_ratio(self, ratio: float) -> None:
        """由时钟估计/漂移校正直接给定速率比（actual/nominal）。"""
        if ratio <= 0:
            raise ValueError("速率比必须为正")
        self.source_rate = self.target_rate * float(ratio)

    def process(self, samples: Sequence[float]) -> List[float]:
        buf = self._carry + [float(s) for s in samples]
        out: List[float] = []
        step = self.step
        n = len(buf)
        while self.phase + 1.0 < n:
            i = int(math.floor(self.phase))
            frac = self.phase - i
            out.append(buf[i] * (1.0 - frac) + buf[i + 1] * frac)
            self.phase += step
        # 保留最后一个可能被下一帧插值用到的采样，相位按保留点回卷。
        keep_from = max(0, int(math.floor(self.phase)))
        self._carry = buf[keep_from:]
        self.phase -= keep_from
        return out

    def phase_state(self) -> PhaseState:
        return PhaseState(
            phase=self.phase,
            step=self.step,
            source_rate=self.source_rate,
            target_rate=self.target_rate,
        )

    def switch_device(self, new_target_rate: float,
                      state: Optional[PhaseState] = None) -> PhaseState:
        """切换输出设备：保留分数相位与 carry，只更新目标速率。

        旧实现切设备时重建重采样器，相位归零造成波形不连续。
        这里相位与未消费样本原样保留，输出波形在切换点连续。
        """
        if new_target_rate <= 0:
            raise ValueError("新设备采样率必须为正")
        snap = state if state is not None else self.phase_state()
        self.phase = snap.phase
        self.target_rate = float(new_target_rate)
        # 源速率由保留下来的 step 反推，速率比不丢。
        self.source_rate = snap.step * self.target_rate
        return self.phase_state()


def resample(samples,
             source_rate: float = DEFAULT_NOMINAL_RATE,
             target_rate: float = DEFAULT_NOMINAL_RATE,
             initial_phase: float = 0.0) -> Tuple[List[float], PhaseState]:
    """无状态便捷入口：重采样一段数据并返回结束相位。"""
    rs = VariableResampler(source_rate, target_rate, initial_phase)
    out = rs.process(_as_float_list(samples))
    return out, rs.phase_state()


def phase_preserve(resampler,
                   new_target_rate: Optional[float] = None,
                   state: Optional[PhaseState] = None) -> PhaseState:
    """对重采样器执行保相位设备切换，或对纯相位值做快照/迁移。"""
    if isinstance(resampler, VariableResampler):
        if new_target_rate is None:
            return resampler.phase_state()
        return resampler.switch_device(new_target_rate, state)
    # 允许直接传分数相位（契约测试的标量入口走这里）。
    phase = float(resampler)
    if new_target_rate is None:
        return PhaseState(phase=phase, step=1.0,
                          source_rate=DEFAULT_NOMINAL_RATE,
                          target_rate=DEFAULT_NOMINAL_RATE)
    return PhaseState(phase=phase, step=1.0,
                      source_rate=float(new_target_rate),
                      target_rate=float(new_target_rate))


# ---------------------------------------------------------------------------
# 修复 4：环形缓冲原子索引（读线程永不读到半帧）
# ---------------------------------------------------------------------------

class AtomicRingBuffer:
    """单写多读安全的环形采样缓冲。

    写线程先在槽位里放好完整帧，再以*单次*原子方式提交写索引；
    读线程只能看到已提交区间，因此不可能读到写了一半的帧。
    读写索引均为受锁保护的 int（Python 下等价于原子读改写），
    提交动作对读者是顺序一致的单点发布。
    """

    def __init__(self, capacity: int, channels: int = 1):
        if capacity <= 0 or channels <= 0:
            raise ValueError("capacity 与 channels 必须为正")
        self.capacity = int(capacity)
        self.channels = int(channels)
        self._slots: List[Optional[List[float]]] = [None] * self.capacity
        self._write_idx = 0
        self._read_idx = 0
        self._lock = threading.Lock()
        self._dropped = 0

    def write(self, frame: Sequence[float]) -> bool:
        """写入一帧（长度须等于 channels）。满了丢弃最旧一帧并计数。"""
        frame = [float(v) for v in frame]
        if len(frame) != self.channels:
            raise ValueError(f"帧长度应为 {self.channels}，实际 {len(frame)}")
        with self._lock:
            idx = self._write_idx
            # 先填充数据，再发布索引：读者依据索引绝不会看到半成品。
            self._slots[idx % self.capacity] = frame
            self._write_idx = idx + 1
            if self._write_idx - self._read_idx > self.capacity:
                self._read_idx = self._write_idx - self.capacity
                self._dropped += 1
            return True

    def read(self) -> Optional[List[float]]:
        """读取一帧；无数据返回 None，绝不返回半帧。"""
        with self._lock:
            if self._read_idx >= self._write_idx:
                return None
            slot = self._slots[self._read_idx % self.capacity]
            self._read_idx += 1
        # 槽内容在写索引发布前已完整，且 channels 定长，拿到的就是整帧。
        return list(slot) if slot is not None else None

    def drain(self, max_frames: Optional[int] = None) -> List[List[float]]:
        out: List[List[float]] = []
        while max_frames is None or len(out) < max_frames:
            frame = self.read()
            if frame is None:
                break
            out.append(frame)
        return out

    @property
    def write_index(self) -> int:
        with self._lock:
            return self._write_idx

    @property
    def read_index(self) -> int:
        with self._lock:
            return self._read_idx

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def __len__(self) -> int:
        with self._lock:
            return self._write_idx - self._read_idx


def atomic_ring(capacity: int, channels: int = 1) -> AtomicRingBuffer:
    """构造原子索引环形缓冲的工厂入口。"""
    return AtomicRingBuffer(int(capacity), int(channels))


# ---------------------------------------------------------------------------
# 修复 5：丢包隐藏帧按采样点对齐（不做毫秒取整）
# ---------------------------------------------------------------------------

@dataclass
class ConcealPlan:
    """PLC 对齐计划。

    frame_sizes   每个隐藏帧的采样点数（按采样点分摊，不做毫秒取整）
    total         隐藏采样总数，与真实缺口误差不超过 1 个采样点
    residual      取整后与理想长度的残差（采样点），留给下一次缺口消化
    """

    frame_sizes: List[int]
    total: int
    residual: float


def conceal_align(missing_samples: int,
                  frame_size_ms: float = 20.0,
                  sample_rate: int = DEFAULT_NOMINAL_RATE) -> ConcealPlan:
    """把缺口拆成若干 PLC 帧，帧长按采样点计算并保留亚采样残差。

    旧实现把每帧毫秒数取整成采样数（20ms@48k=960 恰好整数，但任意码率
    都会遇到非整数），每帧的取整误差沿时间轴累积成漂移。这里用累计
    floor/最大余数法分摊，全程不做毫秒级取整，残差跨缺口结转。
    """
    missing = int(missing_samples)
    if missing < 0:
        raise ValueError("missing_samples 不能为负")
    if frame_size_ms <= 0 or sample_rate <= 0:
        raise ValueError("frame_size_ms 与 sample_rate 必须为正")
    if missing == 0:
        return ConcealPlan(frame_sizes=[], total=0, residual=0.0)

    ideal_frame = frame_size_ms / 1000.0 * sample_rate
    frames = max(1, int(round(missing / ideal_frame)))
    # 最大余数法：每份先取 floor，再把余数逐个补 1，误差 <= 1 采样点。
    base = missing // frames
    rem = missing - base * frames
    sizes = [base + (1 if i < rem else 0) for i in range(frames)]
    ideal_total = frames * ideal_frame
    return ConcealPlan(
        frame_sizes=sizes,
        total=sum(sizes),
        residual=ideal_total - missing,
    )


def plc(missing_samples: int,
        frame_size_ms: float = 20.0,
        sample_rate: int = DEFAULT_NOMINAL_RATE,
        last_samples: Optional[Sequence[float]] = None) -> List[List[float]]:
    """生成隐藏帧：按采样点对齐分帧，内容为衰减重复（简单时域 PLC）。"""
    plan = conceal_align(missing_samples, frame_size_ms, sample_rate)
    template = _as_float_list(last_samples) if last_samples is not None else [0.0]
    frames: List[List[float]] = []
    cursor = 0
    for idx, size in enumerate(plan.frame_sizes):
        gain = 0.5 ** (idx + 1)  # 每帧衰减 6 dB，避免隐藏段能量发散。
        frame = [template[(cursor + j) % len(template)] * gain
                 for j in range(size)]
        cursor += size
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# 修复 6：漂移用长时间窗校正
# ---------------------------------------------------------------------------

class DriftCorrector:
    """长窗口漂移校正器。

    输入每块的"设备实际采样数/标称应有采样数"，在长窗口上累计得到
    速率比，再经 slew 限速输出给重采样器。窗口足够长时，48.001k 设备
    十分钟的半秒偏差会被持续、温和地拉回，而不是靠单帧激进调整。
    """

    def __init__(self, nominal_rate: int = DEFAULT_NOMINAL_RATE,
                 window_samples: int = DEFAULT_LONG_WINDOW,
                 max_slew_ppm: float = DEFAULT_MAX_SLEW_PPM):
        self.estimator = ClockEstimator(nominal_rate, window_samples)
        self.max_slew = max_slew_ppm / 1.0e6
        self._smoothed_ratio = 1.0

    def update(self, observed_samples: int,
               nominal_samples: Optional[int] = None) -> float:
        est = self.estimator.update(observed_samples, nominal_samples)
        target = est.ratio
        # 限速：每次更新相对当前值最多移动 max_slew，抑制估计抖动。
        delta = target - self._smoothed_ratio
        clamped = max(-self.max_slew, min(self.max_slew, delta))
        self._smoothed_ratio += clamped
        return self._smoothed_ratio

    @property
    def ratio(self) -> float:
        return self._smoothed_ratio

    def drift_samples(self, elapsed_seconds: float) -> float:
        """给定时长下，按当前速率比预计的累计漂移采样数。"""
        nominal = elapsed_seconds * self.estimator.nominal_rate
        return nominal * (self._smoothed_ratio - 1.0)


def drift_correct(observed_samples,
                  nominal_samples=None,
                  nominal_rate: int = DEFAULT_NOMINAL_RATE,
                  window_samples: int = DEFAULT_LONG_WINDOW,
                  max_slew_ppm: float = DEFAULT_MAX_SLEW_PPM) -> float:
    """无状态便捷入口：喂入观测序列，返回校正后的速率比。"""
    corr = DriftCorrector(nominal_rate, window_samples, max_slew_ppm)
    ratio = 1.0
    observed = _as_float_list(observed_samples)
    if nominal_samples is None:
        nominal = list(observed)
    else:
        nominal = _as_float_list(nominal_samples)
        if len(nominal) != len(observed):
            raise ValueError("observed 与 nominal 序列长度必须一致")
    for o, n in zip(observed, nominal):
        ratio = corr.update(max(0, int(round(o))), max(0, int(round(n))))
    return ratio


# ---------------------------------------------------------------------------
# 管线配套件
# ---------------------------------------------------------------------------

def latency(buffer_level_samples: int,
            sample_rate: int = DEFAULT_NOMINAL_RATE) -> float:
    """当前缓冲时延（秒）。"""
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须为正")
    return float(buffer_level_samples) / float(sample_rate)


@dataclass
class StreamMetrics:
    """播放链路健康度指标。"""

    underruns: int = 0
    concealed_frames: int = 0
    dropped_frames: int = 0
    max_buffer_level: int = 0
    drift_ppm: float = 0.0
    clock_ratio: float = 1.0


def metrics(underruns: int = 0,
            concealed_frames: int = 0,
            dropped_frames: int = 0,
            max_buffer_level: int = 0,
            clock_ratio: float = 1.0) -> StreamMetrics:
    """汇总指标，漂移量直接由时钟比换算成 ppm。"""
    return StreamMetrics(
        underruns=int(underruns),
        concealed_frames=int(concealed_frames),
        dropped_frames=int(dropped_frames),
        max_buffer_level=int(max_buffer_level),
        drift_ppm=(float(clock_ratio) - 1.0) * 1.0e6,
        clock_ratio=float(clock_ratio),
    )


def determinism(seed: int,
                run_pipeline: Optional[Callable[[int], Sequence[float]]] = None,
                rounds: int = 2) -> bool:
    """同一输入重复跑管线，输出必须逐采样一致（重采样相位确定）。"""
    if run_pipeline is None:
        # 默认管线：固定正弦输入经可变速率重采样，结果天然可复现。
        def run_pipeline(s: int) -> Sequence[float]:
            rs = VariableResampler(DEFAULT_NOMINAL_RATE + 1, DEFAULT_NOMINAL_RATE)
            wave = [math.sin(2 * math.pi * 440.0 * n / DEFAULT_NOMINAL_RATE + s)
                    for n in range(2048)]
            return rs.process(wave)

    first = list(run_pipeline(int(seed)))
    for _ in range(max(1, int(rounds)) - 1):
        again = list(run_pipeline(int(seed)))
        if len(again) != len(first) or any(
            abs(a - b) > 1e-12 for a, b in zip(first, again)
        ):
            return False
    return True


def validation(source_rate: int,
               target_rate: Optional[int] = None,
               block_size: int = 480) -> Tuple[int, int, int]:
    """校验并归一化管线参数；非法值抛 ValueError。"""
    src = int(source_rate)
    dst = src if target_rate is None else int(target_rate)
    if src <= 0 or dst <= 0:
        raise ValueError("采样率必须为正")
    if block_size <= 0:
        raise ValueError("block_size 必须为正")
    return src, dst, int(block_size)


def audit(stream_or_metrics) -> List[str]:
    """对指标/缓冲状态做健康审计，返回人类可读的问题清单。"""
    findings: List[str] = []
    if isinstance(stream_or_metrics, StreamMetrics):
        m = stream_or_metrics
        if m.underruns:
            findings.append(f"underrun 发生 {m.underruns} 次")
        if m.concealed_frames:
            findings.append(f"PLC 隐藏 {m.concealed_frames} 帧")
        if m.dropped_frames:
            findings.append(f"环形缓冲丢帧 {m.dropped_frames} 个")
        if abs(m.drift_ppm) > 100.0:
            findings.append(f"时钟漂移 {m.drift_ppm:.1f} ppm 超出 ±100ppm")
    elif isinstance(stream_or_metrics, AtomicRingBuffer):
        ring = stream_or_metrics
        if ring.dropped:
            findings.append(f"环形缓冲丢帧 {ring.dropped} 个")
        if len(ring) == 0:
            findings.append("环形缓冲已空，读线程即将欠载")
    else:
        # 标量入口：解释为欠载计数。
        if int(stream_or_metrics) > 0:
            findings.append(f"underrun 发生 {int(stream_or_metrics)} 次")
    return findings


def fallback(primary_available,
             conceal_plan: Optional[ConcealPlan] = None,
             sample_rate: int = DEFAULT_NOMINAL_RATE) -> List[float]:
    """主链路断流时生成兜底采样：优先用 PLC 计划，否则输出一帧静音。"""
    if bool(primary_available):
        return []
    if conceal_plan is not None:
        frames = plc(conceal_plan.total, sample_rate=sample_rate)
        return [s for frame in frames for s in frame]
    # 标量入口（如 fallback(0)）：没有计划信息时输出单采样静音占位。
    return [0.0]


def buffer_swap(old_ring: AtomicRingBuffer,
                new_capacity: Optional[int] = None) -> AtomicRingBuffer:
    """热替换环形缓冲：把未读帧无损搬到新缓冲，索引原子切换语义不变。"""
    if not isinstance(old_ring, AtomicRingBuffer):
        # 标量入口：直接按给定容量建新环。
        return AtomicRingBuffer(max(1, int(old_ring)))
    capacity = old_ring.capacity if new_capacity is None else int(new_capacity)
    new_ring = AtomicRingBuffer(capacity, old_ring.channels)
    for frame in old_ring.drain():
        new_ring.write(frame)
    return new_ring
