"""Tests for SSB transmission system."""

import numpy as np

from shannon_bench.simulator.ssb_system import SSBReceiver, SSBTransmitter


class TestSSBSystem:
  """Tests for SSB transmitter and receiver."""

  def test_usb_modulation(self) -> None:
    """Test Upper Sideband (USB) modulation."""
    # Input: 1kHz sine wave at 48kHz sample rate
    fs = 48000
    duration = 0.1
    t = np.arange(int(fs * duration)) / fs
    freq = 1000.0
    audio = np.cos(2 * np.pi * freq * t).astype(np.float32)

    tx = SSBTransmitter(bandwidth_hz=3000, output_sample_rate=fs, mode="USB")
    modulated = tx.modulate(audio, fs)

    # USB of cos(wt) should be exp(jwt) (analytic signal)
    # Check frequency content
    fft = np.fft.fft(modulated)
    freqs = np.fft.fftfreq(len(modulated), 1 / fs)

    # Find peak frequency
    peak_idx = np.argmax(np.abs(fft))
    peak_freq = freqs[peak_idx]

    # Should be positive 1000Hz
    assert np.isclose(peak_freq, 1000.0, atol=10.0)

    # Check that negative frequency component is suppressed
    neg_idx = np.argmin(np.abs(freqs - (-1000.0)))
    assert np.abs(fft[neg_idx]) < np.abs(fft[peak_idx]) * 0.01  # >40dB suppression

  def test_lsb_modulation(self) -> None:
    """Test Lower Sideband (LSB) modulation."""
    # Input: 1kHz sine wave
    fs = 48000
    duration = 0.1
    t = np.arange(int(fs * duration)) / fs
    freq = 1000.0
    audio = np.cos(2 * np.pi * freq * t).astype(np.float32)

    tx = SSBTransmitter(bandwidth_hz=3000, output_sample_rate=fs, mode="LSB")
    modulated = tx.modulate(audio, fs)

    # LSB of cos(wt) should be exp(-jwt) (complex conjugate of analytic)
    # Check frequency content
    fft = np.fft.fft(modulated)
    freqs = np.fft.fftfreq(len(modulated), 1 / fs)

    # Find peak frequency
    peak_idx = np.argmax(np.abs(fft))
    peak_freq = freqs[peak_idx]

    # Should be negative 1000Hz
    assert np.isclose(peak_freq, -1000.0, atol=10.0)

  def test_loopback(self) -> None:
    """Test perfect loopback (Tx -> Rx)."""
    fs = 48000
    duration = 0.1
    t = np.arange(int(fs * duration)) / fs
    # Use a frequency within the passband (300-3300Hz)
    freq = 1000.0
    audio = np.cos(2 * np.pi * freq * t).astype(np.float32)

    tx = SSBTransmitter(bandwidth_hz=3000, output_sample_rate=fs, mode="USB")
    rx = SSBReceiver(output_sample_rate=fs, mode="USB")

    modulated = tx.modulate(audio, fs)
    demodulated = rx.demodulate(modulated, fs)

    # Check correlation
    # Ignore startup transient (filter delay)
    skip = 1000
    corr = np.corrcoef(audio[skip:], demodulated[skip:])[0, 1]

    assert corr > 0.99

  def test_bandwidth_limiting(self) -> None:
    """Test that out-of-band signals are attenuated."""
    fs = 48000
    duration = 0.1
    t = np.arange(int(fs * duration)) / fs
    # 100Hz is below the 300Hz cutoff
    freq = 100.0
    audio = np.cos(2 * np.pi * freq * t).astype(np.float32)

    tx = SSBTransmitter(bandwidth_hz=3000, output_sample_rate=fs, mode="USB")
    modulated = tx.modulate(audio, fs)

    # Check power
    input_power = np.mean(audio**2)
    output_power = np.mean(np.abs(modulated) ** 2)

    # Should be significantly attenuated
    assert output_power < input_power * 0.1
