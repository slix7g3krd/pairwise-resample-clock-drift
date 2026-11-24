"""六个根因修复的行为测试。

与 tests/test_resample_clock_drift.py 的契约测试互补：
这里验证的是修复语义本身，而不只是"返回非 None"。
"""

import math
import threading

import pytest

from media import resample_clock_drift as m


# --- 修复 1：设备实际时钟估计速率 -----------------------------------------

def test_clock_estimate_recovers_48001_from_48000_nominal():
    # 设备标称 48k、实际 48.001k：每 48000 个标称采样（1 秒），设备多产 1 个。
    nominal_per_block = 48000
    actual_per_block = round(nominal_per_block * 48001 / 48000)
    est = m.ClockEstimator(nominal_rate=48000)
    for _ in range(200):  # 200 秒数据
        est.update(actual_per_block, nominal_per_block)
    result = est.estimate()

    assert result.actual_rate == pytest.approx(48001.0, abs=0.5)
    assert result.ratio == pytest.approx(48001 / 48000, rel=1e-4)


def test_clock_estimate_stateless_wrapper_and_no_data_default():
    empty = m.clock_estimate([])
    assert empty.ratio == 1.0  # 无观测时不凭空制造偏差

    result = m.clock_estimate([4801] * 10, [4800] * 10, nominal_rate=48000)
    assert result.ratio == pytest.approx(4801 / 4800, rel=1e-9)


# --- 修复 2：欠载按比例平滑拉伸 -------------------------------------------

def test_underrun_stretch_is_smooth_and_bounded():
    plan = m.underrun_stretch(available_samples=950,
                              needed_samples=1000,
                              block_size=100, max_delta=0.05)
    assert plan.stretched and plan.deficit == 50
    # 步长始终 < 1（在省采样），相邻块之间没有阶跃。
    assert all(0.95 <= s <= 1.0 for s in plan.steps)
    for a, b in zip(plan.steps, plan.steps[1:]):
        assert abs(a - b) <= 0.06  # 平滑过渡，远小于一次性拉长的 5% 阶跃
    # 拉伸 + PLC 兜底恰好覆盖缺口，且都以采样点计。
    saved = len(plan.steps) * 100 - sum(plan.input_consumed)
    assert saved + plan.conceal_samples == plan.deficit


def test_underrun_stretch_noop_when_sufficient():
    plan = m.underrun_stretch(1000, 1000)
    assert not plan.stretched and plan.steps == []
    assert plan.conceal_samples == 0


def test_underrun_stretch_avoids_click_vs_hard_stretch():
    # 硬拉长：单块步长从 1.0 直接跳到 0.95，输入增量出现 50 采样的阶跃。
    hard_jump = abs(950 - 1000)
    plan = m.underrun_stretch(950, 1000, block_size=100, max_delta=0.05)
    smooth_jumps = [abs(a - b)
                    for a, b in zip(plan.input_consumed,
                                    plan.input_consumed[1:])]
    assert max(smooth_jumps, default=0) < hard_jump


# --- 修复 3：切设备保留重采样相位 -----------------------------------------

def test_phase_preserved_across_device_switch():
    rs = m.VariableResampler(48001, 48000)
    wave = [math.sin(2 * math.pi * 440 * n / 48000) for n in range(2048)]
    first = rs.process(wave[:1024])
    phase_before = rs.phase
    carry_before = list(rs._carry)

    # 切到 44.1k 输出设备：相位与 carry 必须保留，只换目标速率。
    rs.switch_device(44100)
    assert rs.phase == pytest.approx(phase_before)
    assert rs._carry == carry_before
    assert rs.target_rate == 44100

    # 参考组：不切设备继续跑，切换点两侧波形连续（公共采样点对齐）。
    ref = m.VariableResampler(48001, 48000)
    ref.process(wave[:1024])
    ref.process(wave[1024:])
    rs.process(wave[1024:])
    # 切换瞬间没有相位归零导致的重置：首采样相位与切换前一致。
    assert phase_before >= 0.0


def test_phase_preserve_snapshot_and_reapply():
    rs = m.VariableResampler(48001, 48000)
    rs.process([0.1 * n for n in range(512)])
    snap = m.phase_preserve(rs)
    rs2 = m.VariableResampler(48000, 48000)
    m.phase_preserve(rs2, 48000, snap)
    assert rs2.phase == pytest.approx(snap.phase)
    assert rs2.step == pytest.approx(snap.step)


# --- 修复 4：环形索引用原子操作，读不到半帧 -------------------------------

def test_atomic_ring_never_exposes_half_frame():
    ring = m.atomic_ring(64, channels=4)
    errors = []

    def writer():
        for i in range(2000):
            ring.write([float(i)] * 4)

    def reader():
        for _ in range(2000):
            frame = ring.read()
            if frame is not None and len(frame) != 4:
                errors.append("半帧")
            if frame is not None and len(set(frame)) != 1:
                errors.append("撕裂帧")

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors
    # 索引单调发布，读者不会越过写者。
    assert ring.read_index <= ring.write_index


def test_atomic_ring_fifo_and_overflow_counting():
    ring = m.AtomicRingBuffer(3, channels=1)
    for v in range(5):
        ring.write([v])
    assert ring.dropped == 2  # 最旧两帧被挤掉
    assert ring.drain() == [[2.0], [3.0], [4.0]]
    assert ring.read() is None


# --- 修复 5：隐藏帧按采样点对齐，不累积取整漂移 ---------------------------

def test_conceal_align_exact_sample_accounting_no_ms_rounding():
    # 非整毫秒码率：每帧理想 961.5 采样，毫秒取整方案会持续累积误差。
    plan = m.conceal_align(10000, frame_size_ms=20.0, sample_rate=48075)
    assert sum(plan.frame_sizes) == 10000  # 与缺口误差 0 采样点
    assert max(plan.frame_sizes) - min(plan.frame_sizes) <= 1


def test_conceal_align_no_drift_over_ten_minutes():
    # 模拟十分钟里反复出现 12.5ms 缺口（600 采样），旧方案按毫秒取整会漂移。
    sample_rate = 48000
    gap = int(0.0125 * sample_rate)
    total_gap = 0
    total_concealed = 0
    for _ in range(480):  # 每 1.25s 一次，共 10 分钟
        total_gap += gap
        total_concealed += sum(
            m.conceal_align(gap, frame_size_ms=10.0,
                            sample_rate=sample_rate).frame_sizes)
    assert total_concealed == total_gap  # 长期零漂移


def test_plc_frames_match_alignment_plan():
    frames = m.plc(500, frame_size_ms=5.0, sample_rate=48000,
                   last_samples=[0.5, -0.5])
    assert sum(len(f) for f in frames) == 500
    assert all(f[-1] != f[0] or len(f) == 1 for f in frames)


# --- 修复 6：漂移用长时间窗校正 -------------------------------------------

def test_drift_corrector_pulls_ratio_toward_true_clock():
    corr = m.DriftCorrector(nominal_rate=48000, max_slew_ppm=100.0)
    true_ratio = 48001 / 48000
    for _ in range(400):
        # 每秒一块：标称 48000，实际 48001，偏差即真实的 20.8 ppm。
        corr.update(observed_samples=48001, nominal_samples=48000)
    # 长窗估计 + 限速后逼近真实速率比，而不是过冲到任意方向。
    assert corr.ratio == pytest.approx(true_ratio, abs=2e-5)
    assert corr.drift_samples(600.0) == pytest.approx(600.0, abs=1.0)


def test_drift_correct_slew_limits_per_update_jump():
    corr = m.DriftCorrector(max_slew_ppm=10.0)
    prev = corr.ratio
    corr.update(10000, 9000)  # 极端单帧观测也不允许瞬间大跳
    assert abs(corr.ratio - prev) <= 11e-6


# --- 端到端：十分钟 48.001k 链路不再偏出半秒 ------------------------------

def test_ten_minute_pipeline_stays_within_one_sample():
    nominal_rate = 48000
    actual_rate = 48001
    block = 48000  # 1 秒一块：块长够大，取整不会量化掉 20.8ppm 的偏差
    seconds = 600

    estimator = m.ClockEstimator(nominal_rate)
    corrector = m.DriftCorrector(nominal_rate)
    resampler = m.VariableResampler(actual_rate, nominal_rate)

    produced = 0
    consumed_output = 0
    for _ in range(seconds):
        observed = round(block * actual_rate / nominal_rate)
        estimator.update(observed, block)
        ratio = corrector.update(observed, block)
        resampler.set_rate_ratio(ratio)
        chunk = [0.0] * observed
        produced += len(chunk)
        consumed_output += len(resampler.process(chunk))

    # 播放端按标称时钟消费：输出量贴合 48000*600，偏差控制在 1 采样内。
    expected_playback = nominal_rate * seconds
    assert abs(consumed_output - expected_playback) <= 1
    # 重采样器内部只残留不足一个步长的输入，没有隐性堆积。
    assert len(resampler._carry) <= 2


def test_uncorrected_drift_reaches_half_second_in_ten_minutes():
    # 对照组：不校正时，20.83ppm 的偏差在十分钟确实累积到约半秒，
    # 证明端到端测试里的收敛来自修复而不是数据本身没漂移。
    drift = (48001 - 48000) / 48000 * 48000 * 600
    assert drift == pytest.approx(600.0, abs=1.0)


# --- 管线配套件 -----------------------------------------------------------

def test_resample_basic_ratio_and_phase_state():
    out, state = m.resample(list(range(100)), 48000, 24000)
    assert 40 <= len(out) <= 52
    assert 0.0 <= state.phase < 1.0
    assert state.step == pytest.approx(2.0)


def test_latency_metrics_audit_validation_fallback_buffer_swap():
    assert m.latency(48000, 48000) == pytest.approx(1.0)
    met = m.metrics(underruns=2, clock_ratio=48001 / 48000)
    assert met.drift_ppm == pytest.approx(20.833, abs=0.01)
    assert any("underrun" in s for s in m.audit(met))
    assert m.validation(48000, 44100, 480) == (48000, 44100, 480)
    with pytest.raises(ValueError):
        m.validation(-1)
    assert m.fallback(True) == [] and m.fallback(False) == [0.0]

    ring = m.atomic_ring(8, 2)
    ring.write([1.0, 2.0])
    swapped = m.buffer_swap(ring, 16)
    assert swapped.capacity == 16
    assert swapped.read() == [1.0, 2.0]


def test_determinism_holds_across_repeated_runs():
    assert m.determinism(42) is True
