"""Single Sideband (SSB) transmission system implementation.

This module implements the Transmitter and Receiver classes for SSB modulation,
supporting both Upper Sideband (USB) and Lower Sideband (LSB).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.signal

from shannon_bench.simulator.transmission_system import Receiver, Transmitter


class SSBTransmitter(Transmitter):
  """Single Sideband (SSB) Transmitter.

  Implements SSB modulation using the Hilbert transform method.
  Supports both USB and LSB modes.
  """

  def __init__(
    self,
    bandwidth_hz: float = 3000.0,
    output_sample_rate: int = 48000,
    mode: Literal["USB", "LSB"] = "USB",
  ) -> None:
    """Initialize SSB transmitter.

    Args:
      bandwidth_hz: Signal bandwidth in Hz.
      output_sample_rate: Output sample rate in Hz.
      mode: Sideband mode ("USB" or "LSB").
    """
    self._bandwidth_hz = bandwidth_hz
    self._output_sample_rate = output_sample_rate
    self.mode = mode

  @property
  def name(self) -> str:
    return f"SSB_{self.mode}_Tx"

  @property
  def bandwidth_hz(self) -> float:
    return self._bandwidth_hz

  @property
  def output_sample_rate(self) -> int:
    return self._output_sample_rate

  def modulate(
    self, audio: npt.NDArray[np.float32], input_sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Modulate audio to SSB baseband signal.

    Args:
      audio: Input audio samples (mono, float32).
      input_sample_rate: Input audio sample rate.

    Returns:
      Complex baseband signal (I/Q).
    """
    # 1. Resample if necessary
    if input_sample_rate != self.output_sample_rate:
      # Use polyphase resampling for better quality (anti-aliasing)
      # Find integer ratio if possible, or use large integers
      gcd = np.gcd(input_sample_rate, self.output_sample_rate)
      up = self.output_sample_rate // gcd
      down = input_sample_rate // gcd
      audio = scipy.signal.resample_poly(audio, up, down)

    # 2. Bandpass filter audio (e.g., 300-2700 Hz for voice) to limit bandwidth
    # We'll use a simple Butterworth filter.
    # Note: For a generic SSB transmitter, we might want to make these cutoffs
    # configurable.
    # For now, we'll assume standard voice SSB bandwidth if not specified,
    # but strictly speaking, the 'bandwidth_hz' property is just for reporting.
    # Let's implement a filter that respects the bandwidth.
    # Assuming voice, we usually want 300Hz to (300 + bandwidth) Hz.
    nyquist = self.output_sample_rate / 2
    low_cut = 300.0
    high_cut = min(low_cut + self.bandwidth_hz, nyquist - 100)

    sos = scipy.signal.butter(
      N=4,
      Wn=[low_cut, high_cut],
      btype="bandpass",
      fs=self.output_sample_rate,
      output="sos",
    )
    filtered_audio = scipy.signal.sosfiltfilt(sos, audio)

    # 3. Hilbert transform for analytic signal
    analytic_signal = scipy.signal.hilbert(filtered_audio)

    # 4. Select sideband
    if self.mode == "USB":
      # USB is the analytic signal (positive frequencies)
      return analytic_signal.astype(np.complex64)
    # LSB is the complex conjugate (negative frequencies)
    return np.conj(analytic_signal).astype(np.complex64)


class SSBReceiver(Receiver):
  """Single Sideband (SSB) Receiver.

  Implements SSB demodulation (coherent detection).
  """

  def __init__(
    self,
    output_sample_rate: int = 48000,
    mode: Literal["USB", "LSB"] = "USB",
  ) -> None:
    """Initialize SSB receiver.

    Args:
      output_sample_rate: Output audio sample rate in Hz.
      mode: Sideband mode ("USB" or "LSB").
    """
    self._output_sample_rate = output_sample_rate
    self.mode = mode

  @property
  def name(self) -> str:
    return f"SSB_{self.mode}_Rx"

  @property
  def output_sample_rate(self) -> int:
    return self._output_sample_rate

  def demodulate(
    self, signal: npt.NDArray[np.complex64], input_sample_rate: int
  ) -> npt.NDArray[np.float32]:
    """Demodulate SSB baseband signal to audio.

    Args:
      signal: Complex baseband signal.
      input_sample_rate: Input signal sample rate.

    Returns:
      Demodulated audio samples.
    """
    # 1. Demodulate (Real part extraction)
    # For baseband SSB, the message is in the real part.
    # If we had frequency offsets, we'd need to correct them first,
    # but the ChannelSimulator handles offsets by shifting the signal.
    # However, if the channel added a frequency offset, the signal is now
    # signal * exp(j * 2pi * offset * t).
    # Taking the real part of that mixes the I and Q components.
    # Ideally, a receiver would have a carrier recovery loop (PLL) or manual tuning.
    # For this simple simulation, we assume perfect synchronization or that
    # the user wants to hear the "Donald Duck" effect of frequency offset.
    # So we just take the real part.

    # Wait, if we just take the real part of a frequency-shifted SSB signal,
    # we get the frequency-shifted audio (Donald Duck effect).
    # That is exactly what we want to simulate for "impaired channel".

    audio = np.real(signal)

    # 2. Resample if necessary
    if input_sample_rate != self.output_sample_rate:
      # Use polyphase resampling
      gcd = np.gcd(input_sample_rate, self.output_sample_rate)
      up = self.output_sample_rate // gcd
      down = input_sample_rate // gcd
      audio = scipy.signal.resample_poly(audio, up, down)

    # 3. Optional: Low-pass filter to remove out-of-band noise
    # (The ear acts as a filter, but good to clean up)
    nyquist = self.output_sample_rate / 2
    # Assuming standard voice bandwidth ~3kHz
    cutoff = 3000.0
    if cutoff < nyquist:
      sos = scipy.signal.butter(
        N=4, Wn=cutoff, btype="lowpass", fs=self.output_sample_rate, output="sos"
      )
      audio = scipy.signal.sosfiltfilt(sos, audio)

    return audio.astype(np.float32)
