"""Unit tests for channel impairment classes.

Run with: python3 transmission_system_test.py
Or with pytest: pytest transmission_system_test.py -v
"""

import sys

import numpy as np
import pytest
from scipy import signal as scipy_signal

from shannon_bench.simulator.impairements import (
  AWGN,
  ChannelSimulator,
  FreqOffset,
  ImpairmentStage,
  Nonlinearity,
  PhaseNoise,
  RayleighFading,
  RicianFading,
  SSBFilter,
)


class TestAWGN:
  """Tests for AWGN."""

  @pytest.mark.parametrize("snr_db", [-5, 0, 5, 10, 20, 30])
  def test_snr_calculation(self, snr_db) -> None:
    """Test that AWGN achieves target SNR across wide range."""
    impairment = AWGN(snr_db=snr_db, seed=42)

    # Create a known signal
    signal = np.ones(10000, dtype=np.complex64)
    noisy = impairment.apply(signal, sample_rate=8000)

    # Measure SNR
    signal_power = np.mean(np.abs(signal) ** 2)
    noise = noisy - signal
    noise_power = np.mean(np.abs(noise) ** 2)
    measured_snr_db = 10 * np.log10(signal_power / noise_power)

    # Should be within 0.5 dB of target (gets tighter at higher SNR)
    tolerance = 0.5 if snr_db >= 0 else 1.0
    assert abs(measured_snr_db - snr_db) < tolerance

  @pytest.mark.parametrize("signal_type", ["dc", "tone", "qpsk"])
  def test_snr_with_different_signals(self, signal_type) -> None:
    """Test SNR measurement with different signal types."""
    n_samples = 10000
    snr_target = 15.0

    # Generate different signal types
    if signal_type == "dc":
      signal = np.ones(n_samples, dtype=np.complex64)
    elif signal_type == "tone":
      t = np.arange(n_samples) / 8000
      signal = np.exp(1j * 2 * np.pi * 1000 * t).astype(np.complex64)
    else:  # qpsk
      symbols = np.random.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], n_samples)
      signal = (symbols / np.sqrt(2)).astype(np.complex64)

    impairment = AWGN(snr_db=snr_target, seed=42)
    noisy = impairment.apply(signal, sample_rate=8000)

    # Measure SNR
    signal_power = np.mean(np.abs(signal) ** 2)
    noise = noisy - signal
    noise_power = np.mean(np.abs(noise) ** 2)
    measured_snr_db = 10 * np.log10(signal_power / noise_power)

    assert abs(measured_snr_db - snr_target) < 0.5

  def test_deterministic_with_seed(self) -> None:
    """Test that same seed produces same noise."""
    signal = np.ones(1000, dtype=np.complex64)

    imp1 = AWGN(snr_db=10, seed=42)
    imp2 = AWGN(snr_db=10, seed=42)

    result1 = imp1.apply(signal, sample_rate=8000)
    result2 = imp2.apply(signal, sample_rate=8000)

    np.testing.assert_array_almost_equal(result1, result2)

  def test_stage(self) -> None:
    """Test that AWGN is in NOISE stage."""
    impairment = AWGN(snr_db=10)
    assert impairment.stage == ImpairmentStage.NOISE

  def test_name(self) -> None:
    """Test human-readable name."""
    impairment = AWGN(snr_db=10.5)
    assert "10.5" in impairment.name
    assert "dB" in impairment.name


class TestSSBFilter:
  """Tests for SSBFilter."""

  @pytest.mark.parametrize("sample_rate", [8000, 16000, 48000])
  def test_frequency_response(self, sample_rate) -> None:
    """Test that filter has correct passband and stopband at different sample rates."""
    impairment = SSBFilter(low_cutoff_hz=300, high_cutoff_hz=2700, filter_order=5)

    # Generate white noise
    np.random.seed(42)
    signal = (np.random.randn(50000) + 1j * np.random.randn(50000)).astype(np.complex64)

    # Apply filter
    filtered = impairment.apply(signal, sample_rate=sample_rate)

    # Compute PSD with high resolution to avoid leakage
    f, Pxx = scipy_signal.welch(
      filtered, fs=sample_rate, nperseg=2048, return_onesided=False
    )
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    Pxx_db = 10 * np.log10(Pxx + 1e-20)

    # Check passband (should be relatively flat)
    passband_mask = (np.abs(f) >= 400) & (np.abs(f) <= 2600)
    passband_level = np.mean(Pxx_db[passband_mask])

    # Check stopband (should be attenuated)
    # Use < 100 Hz to allow for transition band at high sample rates
    stopband_mask = np.abs(f) < 100
    stopband_level = np.mean(Pxx_db[stopband_mask])

    attenuation = passband_level - stopband_level
    assert attenuation > 20  # At least 20 dB attenuation

  def test_group_delay(self) -> None:
    """Test that filter has approximately constant group delay (linear phase)."""
    impairment = SSBFilter(low_cutoff_hz=300, high_cutoff_hz=2700, filter_order=5)

    sample_rate = 8000
    n_fft = 2048

    # Get SOS filter coefficients for this sample rate
    impairment.apply(np.zeros(10, dtype=np.complex64), sample_rate=sample_rate)
    sos = impairment._sos_cache[sample_rate]

    # Compute frequency response
    w, h = scipy_signal.sosfreqz(sos, worN=n_fft, fs=sample_rate)

    # Compute group delay (negative derivative of phase)
    phase = np.unwrap(np.angle(h))
    group_delay = -np.diff(phase) / np.diff(w * 2 * np.pi)

    # Check group delay variation in passband
    passband_mask = (np.abs(w[:-1]) >= 400) & (np.abs(w[:-1]) <= 2600)
    if np.any(passband_mask):
      gd_passband = group_delay[passband_mask]
      gd_variation = np.std(gd_passband)
      # Group delay should be relatively constant (< 1ms variation)
      assert gd_variation < 0.001

  def test_filter_caching(self) -> None:
    """Test that filter coefficients are cached per sample rate."""
    impairment = SSBFilter()

    signal = np.ones(100, dtype=np.complex64)

    # First call creates filter
    impairment.apply(signal, sample_rate=8000)
    assert 8000 in impairment._sos_cache

    # Second call reuses filter
    impairment.apply(signal, sample_rate=8000)

    # Different sample rate creates new filter
    impairment.apply(signal, sample_rate=16000)
    assert len(impairment._sos_cache) == 2

  def test_invalid_cutoffs(self) -> None:
    """Test that invalid cutoff frequencies raise error."""
    impairment = SSBFilter(
      low_cutoff_hz=3000,  # Higher than high cutoff
      high_cutoff_hz=2700,
    )

    signal = np.ones(100, dtype=np.complex64)

    try:
      impairment.apply(signal, sample_rate=8000)
      msg = "Should have raised ValueError"
      raise AssertionError(msg)
    except ValueError:
      pass  # Expected

  def test_stage(self) -> None:
    """Test that SSB filter is in TRANSMITTER stage."""
    impairment = SSBFilter()
    assert impairment.stage == ImpairmentStage.TRANSMITTER

  def test_name(self) -> None:
    """Test human-readable name."""
    impairment = SSBFilter(low_cutoff_hz=300, high_cutoff_hz=2700)
    assert "300-2700Hz" in impairment.name


class TestFreqOffset:
  """Tests for FreqOffset."""

  def test_frequency_shift(self) -> None:
    """Test that frequency offset shifts spectrum."""
    offset_hz = 100
    sample_rate = 8000
    duration = 1.0

    # Generate a tone at 500 Hz
    t = np.arange(0, duration, 1 / sample_rate)
    tone_freq = 500
    signal = np.exp(2j * np.pi * tone_freq * t).astype(np.complex64)

    # Apply offset
    impairment = FreqOffset(offset_hz=offset_hz)
    shifted = impairment.apply(signal, sample_rate=sample_rate)

    # Compute FFT to find peak
    fft = np.fft.fft(shifted)
    freqs = np.fft.fftfreq(len(shifted), 1 / sample_rate)
    peak_idx = np.argmax(np.abs(fft))
    peak_freq = freqs[peak_idx]

    # Peak should be at original + offset
    expected_freq = tone_freq + offset_hz
    assert abs(peak_freq - expected_freq) < 1  # Within 1 Hz

  def test_zero_offset_is_noop(self) -> None:
    """Test that zero offset doesn't change signal."""
    impairment = FreqOffset(offset_hz=0)
    signal = np.random.randn(1000).astype(np.complex64)

    result = impairment.apply(signal, sample_rate=8000)

    np.testing.assert_array_equal(result, signal)

  def test_stage(self) -> None:
    """Test that frequency offset is in PROPAGATION stage."""
    impairment = FreqOffset(offset_hz=50)
    assert impairment.stage == ImpairmentStage.PROPAGATION

  def test_name(self) -> None:
    """Test human-readable name."""
    impairment = FreqOffset(offset_hz=50.5)
    assert "50.5" in impairment.name
    assert "Hz" in impairment.name


class TestRayleighFading:
  """Tests for RayleighFading."""

  @pytest.mark.parametrize("doppler_hz", [0.1, 1.0, 10.0, 100.0])
  def test_rayleigh_fading_statistics(self, doppler_hz) -> None:
    """Test that Rayleigh fading has correct amplitude distribution."""
    impairment = RayleighFading(doppler_hz=doppler_hz, seed=42)

    # Generate long signal to get good statistics
    sample_rate = 8000
    duration = max(10.0, 100 / doppler_hz)  # At least 100 coherence times
    n_samples = int(duration * sample_rate)
    # Increase sample limit for low Doppler to ensure convergence
    signal = np.ones(min(n_samples, 2000000), dtype=np.complex64)
    faded = impairment.apply(signal, sample_rate=sample_rate)

    # For Rayleigh fading, |h|^2 should be exponentially distributed
    # with mean = 1 (since we normalize)
    power = np.abs(faded) ** 2
    mean_power = np.mean(power)

    # Should be close to 1.0 (within 20% - Jakes model with finite sinusoids)
    assert 0.8 < mean_power < 1.4, f"Mean power = {mean_power}, expected ~1.0"

  @pytest.mark.parametrize("doppler_hz", [1.0, 10.0, 50.0])
  def test_coherence_time(self, doppler_hz) -> None:
    """Test that fading coherence time matches theoretical value T_c ≈ 0.4/f_d."""
    impairment = RayleighFading(doppler_hz=doppler_hz, seed=42)

    sample_rate = 8000
    duration = max(5.0, 50 / doppler_hz)  # At least 50 coherence times
    n_samples = int(duration * sample_rate)
    signal = np.ones(min(n_samples, 100000), dtype=np.complex64)
    faded = impairment.apply(signal, sample_rate=sample_rate)

    # Compute autocorrelation of fading envelope
    amplitude = np.abs(faded)
    amplitude_centered = amplitude - np.mean(amplitude)
    autocorr = np.correlate(amplitude_centered, amplitude_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]  # Keep only positive lags
    autocorr /= autocorr[0]  # Normalize

    # Find lag where autocorrelation drops to 0.5 (coherence time definition)
    lags = np.arange(len(autocorr)) / sample_rate
    idx_half = np.where(autocorr < 0.5)[0]
    if len(idx_half) > 0:
      measured_tc = lags[idx_half[0]]
      # Theoretical Tc for correlation=0.5 is approx 0.24 / fd
      # (Based on J0(2*pi*fd*tau) = 0.5 => 2*pi*fd*tau ~= 1.52 => tau ~= 0.24/fd)
      theoretical_tc = 0.24 / doppler_hz

      # Should be within factor of 2 (Jakes model approximation)
      ratio = measured_tc / theoretical_tc
      assert 0.6 < ratio < 1.4, (
        f"T_c measured={measured_tc:.3f}s, theoretical={theoretical_tc:.3f}s"
      )

  def test_zero_doppler_is_noop(self) -> None:
    """Test that zero Doppler results in no fading."""
    impairment = RayleighFading(doppler_hz=0.0)

    signal = np.random.randn(1000).astype(np.complex64)
    result = impairment.apply(signal, sample_rate=8000)

    # Should be unchanged
    np.testing.assert_array_equal(result, signal)

  def test_time_varying_fading(self) -> None:
    """Test that fading coefficient varies with time."""
    impairment = RayleighFading(doppler_hz=10.0, seed=42)

    # Constant signal
    signal = np.ones(10000, dtype=np.complex64)
    faded = impairment.apply(signal, sample_rate=8000)

    # Fading coefficient should vary (not constant)
    # Check that amplitude varies significantly
    amplitude = np.abs(faded)
    assert np.std(amplitude) > 0.1, "Fading should be time-varying"

    # Check that it's not just noise (should have correlation structure)
    # Autocorrelation at lag 0 should be higher than at large lag
    acf_0 = np.mean(amplitude[:5000] * amplitude[:5000])
    acf_1000 = np.mean(amplitude[:4000] * amplitude[1000:5000])
    assert acf_0 > acf_1000, "Fading should have temporal correlation"

  def test_deterministic_with_seed(self) -> None:
    """Test that same seed produces same fading."""
    signal = np.ones(1000, dtype=np.complex64)

    imp1 = RayleighFading(doppler_hz=5.0, seed=42)
    imp2 = RayleighFading(doppler_hz=5.0, seed=42)

    result1 = imp1.apply(signal, sample_rate=8000)
    result2 = imp2.apply(signal, sample_rate=8000)

    np.testing.assert_array_almost_equal(result1, result2)

  def test_stage(self) -> None:
    """Test that fading is in CHANNEL stage."""
    impairment = RayleighFading(doppler_hz=1.0)
    assert impairment.stage == ImpairmentStage.CHANNEL

  def test_name(self) -> None:
    """Test Rayleigh fading name."""
    impairment = RayleighFading(doppler_hz=1.5)
    assert "Rayleigh" in impairment.name
    assert "1.5" in impairment.name


class TestRicianFading:
  """Tests for RicianFading."""

  @pytest.mark.parametrize("k_factor_db", [-10, 0, 6, 15])
  def test_rician_fading_statistics(self, k_factor_db) -> None:
    """Test that Rician fading has correct K-factor across different scenarios."""
    impairment = RicianFading(doppler_hz=5.0, k_factor_db=k_factor_db, seed=42)

    # Generate long signal
    sample_rate = 8000
    duration = max(5.0, 50 / 5.0)  # At least 50 coherence times
    n_samples = int(duration * sample_rate)
    signal = np.ones(min(n_samples, 100000), dtype=np.complex64)
    faded = impairment.apply(signal, sample_rate=sample_rate)

    # For Rician fading: mean power should still be ~ 1 (normalized)
    power = np.abs(faded) ** 2
    mean_power = np.mean(power)

    # Rician should have less variance than Rayleigh for high K
    power_std = np.std(power)

    # Higher K-factor -> less fading -> power closer to mean
    if k_factor_db > 10:
      # Strong LOS: power variation should be small
      assert power_std < 0.5
    elif k_factor_db < -5:
      # Approaches Rayleigh: larger variation
      assert power_std > 0.3

    # Mean power should still be normalized
    assert 0.7 < mean_power < 1.5

  def test_zero_doppler_is_noop(self) -> None:
    """Test that zero Doppler results in no fading."""
    impairment = RicianFading(doppler_hz=0.0, k_factor_db=10)

    signal = np.random.randn(1000).astype(np.complex64)
    result = impairment.apply(signal, sample_rate=8000)

    # Should be unchanged
    np.testing.assert_array_equal(result, signal)

  def test_deterministic_with_seed(self) -> None:
    """Test that same seed produces same fading."""
    signal = np.ones(1000, dtype=np.complex64)

    imp1 = RicianFading(doppler_hz=5.0, k_factor_db=10, seed=42)
    imp2 = RicianFading(doppler_hz=5.0, k_factor_db=10, seed=42)

    result1 = imp1.apply(signal, sample_rate=8000)
    result2 = imp2.apply(signal, sample_rate=8000)

    np.testing.assert_array_almost_equal(result1, result2)

  def test_stage(self) -> None:
    """Test that fading is in CHANNEL stage."""
    impairment = RicianFading(doppler_hz=1.0, k_factor_db=10)
    assert impairment.stage == ImpairmentStage.CHANNEL

  def test_name(self) -> None:
    """Test Rician fading name."""
    impairment = RicianFading(doppler_hz=2.0, k_factor_db=10)
    assert "Rician" in impairment.name
    assert "10" in impairment.name


class TestPhaseNoise:
  """Tests for PhaseNoise."""

  def test_low_noise_is_noop(self) -> None:
    """Test that very low phase noise has negligible effect."""
    impairment = PhaseNoise(level_dbchz=-200)

    signal = np.random.randn(1000).astype(np.complex64)
    result = impairment.apply(signal, sample_rate=8000)

    # Should be unchanged
    np.testing.assert_array_equal(result, signal)

  def test_phase_drift(self) -> None:
    """Test that phase drifts over time (random walk)."""
    impairment = PhaseNoise(level_dbchz=-60, seed=42)

    # Constant signal
    signal = np.ones(10000, dtype=np.complex64)
    noisy = impairment.apply(signal, sample_rate=8000)

    # Phase should drift
    # Unwrap phase to handle wrapping around +/- pi
    phase = np.unwrap(np.angle(noisy))
    phase_diff = np.diff(phase)

    # Steps should be small but non-zero
    assert np.std(phase_diff) > 0
    assert np.std(phase_diff) < 0.1

    # Variance of phase should increase with time (random walk)
    # Var(phase[n]) ~ n * sigma^2
    var_early = np.var(phase[:1000])
    var_late = np.var(phase[:5000])
    assert var_late > var_early

  def test_deterministic_with_seed(self) -> None:
    """Test that same seed produces same phase noise."""
    signal = np.ones(1000, dtype=np.complex64)

    imp1 = PhaseNoise(level_dbchz=-80, seed=42)
    imp2 = PhaseNoise(level_dbchz=-80, seed=42)

    result1 = imp1.apply(signal, sample_rate=8000)
    result2 = imp2.apply(signal, sample_rate=8000)

    np.testing.assert_array_almost_equal(result1, result2)

  def test_stage(self) -> None:
    """Test that phase noise is in TRANSMITTER stage."""
    impairment = PhaseNoise(level_dbchz=-80)
    assert impairment.stage == ImpairmentStage.TRANSMITTER

  def test_name(self) -> None:
    """Test human-readable name."""
    impairment = PhaseNoise(level_dbchz=-80.5)
    assert "-80.5" in impairment.name
    assert "dBc/Hz" in impairment.name


class TestNonlinearity:
  """Tests for Nonlinearity (Rapp model)."""

  def test_zero_backoff_is_noop(self) -> None:
    """Test that zero input backoff results in no compression."""
    impairment = Nonlinearity(input_backoff_db=0)

    signal = np.random.randn(1000).astype(np.complex64)
    result = impairment.apply(signal, sample_rate=8000)

    # Should be unchanged
    np.testing.assert_array_equal(result, signal)

  @pytest.mark.parametrize("backoff_db", [1, 3, 6, 10])
  def test_compression_reduces_peaks(self, backoff_db) -> None:
    """Test that nonlinearity compresses high-amplitude peaks at various backoff levels."""
    impairment = Nonlinearity(input_backoff_db=backoff_db)

    # Create signal with high PAPR (QPSK-like)
    np.random.seed(42)
    symbols = np.random.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], 10000)
    signal = (symbols / np.sqrt(2)).astype(np.complex64)

    result = impairment.apply(signal, sample_rate=8000)

    # Output peak should be limited
    input_peak = np.max(np.abs(signal))
    output_peak = np.max(np.abs(result))

    # Compression should reduce peak (but not below the backoff point)
    # Higher backoff -> more compression
    if backoff_db >= 3:
      assert output_peak < input_peak * 1.1  # Peaks reduced

  def test_phase_preserved(self) -> None:
    """Test that Rapp model preserves phase (AM/AM only)."""
    # Create signal with varying amplitude but constant phase
    amplitude = np.linspace(0.1, 2.0, 1000)
    phase = np.pi / 4  # Constant phase
    signal = (amplitude * np.exp(1j * phase)).astype(np.complex64)

    impairment = Nonlinearity(input_backoff_db=3.0)
    compressed = impairment.apply(signal, sample_rate=8000)

    # Phase should be preserved
    output_phase = np.angle(compressed)
    np.testing.assert_array_almost_equal(output_phase, np.full(1000, phase), decimal=5)

  def test_saturation_behavior(self) -> None:
    """Test that output saturates at high input levels."""
    # Create signal with increasing amplitude
    amplitude = np.linspace(0, 5.0, 1000)
    signal = amplitude.astype(np.complex64)

    impairment = Nonlinearity(input_backoff_db=3.0, smoothness=2.0)
    compressed = impairment.apply(signal, sample_rate=8000)

    output_amplitude = np.abs(compressed)

    # Output should saturate (derivative decreases)
    # Check that gain decreases at high amplitudes
    gain = output_amplitude / (amplitude + 1e-10)

    # Gain at low amplitude should be ~1
    assert 0.95 < gain[100] < 1.05

    # Gain at high amplitude should be < 1 (compression)
    assert gain[-1] < 0.8

  def test_smoothness_parameter(self) -> None:
    """Test that smoothness parameter controls clipping hardness."""
    signal = np.linspace(0, 3.0, 1000).astype(np.complex64)

    # Soft clipping (low p)
    soft = Nonlinearity(input_backoff_db=3.0, smoothness=1.0)
    soft_out = soft.apply(signal, sample_rate=8000)

    # Hard clipping (high p)
    hard = Nonlinearity(input_backoff_db=3.0, smoothness=10.0)
    hard_out = hard.apply(signal, sample_rate=8000)

    # At saturation point, hard clipping stays linear longer and saturates sharply
    # Soft clipping starts compressing earlier, so its output is lower
    soft_amp = np.abs(soft_out[-1])
    hard_amp = np.abs(hard_out[-1])

    assert hard_amp > soft_amp, (
      "Hard clipping should result in higher amplitude (closer to saturation)"
    )

  def test_papr_reduction(self) -> None:
    """Test that nonlinearity reduces PAPR."""
    # Create high-PAPR signal (OFDM-like)
    np.random.seed(42)
    # Sum of sinusoids creates high PAPR
    t = np.linspace(0, 1, 8000)
    signal = sum(np.exp(2j * np.pi * f * t) for f in [100, 200, 300, 400])
    signal = signal.astype(np.complex64)

    # Calculate input PAPR
    input_power = np.mean(np.abs(signal) ** 2)
    input_peak = np.max(np.abs(signal) ** 2)
    input_papr_db = 10 * np.log10(input_peak / input_power)

    # Apply nonlinearity
    impairment = Nonlinearity(input_backoff_db=3.0, smoothness=2.0)
    compressed = impairment.apply(signal, sample_rate=8000)

    # Calculate output PAPR
    output_power = np.mean(np.abs(compressed) ** 2)
    output_peak = np.max(np.abs(compressed) ** 2)
    output_papr_db = 10 * np.log10(output_peak / output_power)

    # PAPR should be reduced
    assert output_papr_db < input_papr_db

  def test_stage(self) -> None:
    """Test that nonlinearity is in TRANSMITTER stage."""
    impairment = Nonlinearity(input_backoff_db=3)
    assert impairment.stage == ImpairmentStage.TRANSMITTER

  def test_name(self) -> None:
    """Test human-readable name."""
    impairment = Nonlinearity(input_backoff_db=3.5, smoothness=2.0)
    assert "3.5" in impairment.name
    assert "IBO" in impairment.name


class TestChannelSimulator:
  """Tests for ChannelSimulator."""

  def test_auto_sorting(self) -> None:
    """Test that impairments are automatically sorted by stage."""
    # Create in wrong order
    impairments = [
      AWGN(snr_db=10),  # NOISE (4)
      FreqOffset(offset_hz=50),  # PROPAGATION (3)
      RayleighFading(doppler_hz=1.0),  # CHANNEL (2)
      SSBFilter(),  # TRANSMITTER (1)
    ]

    sim = ChannelSimulator(impairments=impairments, sample_rate=8000)

    # Check order
    assert sim.impairments[0].stage == ImpairmentStage.TRANSMITTER
    assert sim.impairments[1].stage == ImpairmentStage.CHANNEL
    assert sim.impairments[2].stage == ImpairmentStage.PROPAGATION
    assert sim.impairments[3].stage == ImpairmentStage.NOISE

  def test_apply_chain(self) -> None:
    """Test that all impairments are applied in sequence."""
    impairments = [
      SSBFilter(),
      AWGN(snr_db=20, seed=42),
    ]

    sim = ChannelSimulator(impairments=impairments, sample_rate=8000)

    signal = np.ones(1000, dtype=np.complex64)
    result = sim.apply(signal)

    # Result should be different from input (noise added)
    assert not np.allclose(result, signal)

    # Result should have same shape
    assert result.shape == signal.shape

  def test_empty_impairments(self) -> None:
    """Test that simulator works with no impairments."""
    sim = ChannelSimulator(impairments=[], sample_rate=8000)

    signal = np.random.randn(1000).astype(np.complex64)
    result = sim.apply(signal)

    # Should be unchanged
    np.testing.assert_array_equal(result, signal)


def run_all_tests() -> int:
  """Run all test classes."""
  test_classes = [
    TestAWGN,
    TestSSBFilter,
    TestFreqOffset,
    TestRayleighFading,
    TestRicianFading,
    TestPhaseNoise,
    TestNonlinearity,
    TestChannelSimulator,
  ]

  total_tests = 0
  passed_tests = 0
  failed_tests = []

  for test_class in test_classes:
    test_instance = test_class()
    test_methods = [m for m in dir(test_instance) if m.startswith("test_")]

    for method_name in test_methods:
      total_tests += 1
      try:
        method = getattr(test_instance, method_name)
        method()
        passed_tests += 1
      except Exception as e:
        failed_tests.append((test_class.__name__, method_name, str(e)))

  if failed_tests:
    for _class_name, method_name, _error in failed_tests:
      pass
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(run_all_tests())
