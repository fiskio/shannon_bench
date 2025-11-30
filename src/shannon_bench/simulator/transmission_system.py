"""Radio transmission system evaluation framework.

This module provides a modular architecture for evaluating radio transmission
systems by simulating the complete TX -> Channel -> RX chain and measuring
Word Error Rate (WER) using ASR models like Whisper.

The architecture consists of:
- Transmitter: Converts audio to modulated RF signal
- ChannelSimulator: Applies realistic channel impairments
- Receiver: Demodulates RF signal back to audio
- TransmissionSystem: Orchestrates the complete chain
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import IntEnum

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


class ImpairmentStage(IntEnum):
  """Defines the canonical order of channel impairments.

  Impairments are applied in this order to match the physical reality
  of a radio transmission chain:
  1. TRANSMITTER: TX-side impairments (filters, PA nonlinearity, TX phase noise)
  2. CHANNEL: Propagation effects (fading, multipath)
  3. PROPAGATION: Carrier effects (frequency offset, Doppler)
  4. NOISE: Additive noise (AWGN) - always applied last
  """

  TRANSMITTER = 1
  CHANNEL = 2
  PROPAGATION = 3
  NOISE = 4


class ChannelImpairment(ABC):
  """Abstract base class for channel impairments.

  Each impairment represents a specific physical effect that can be applied
  to a signal. Impairments are automatically ordered by their stage to ensure
  physically realistic simulation.
  """

  @abstractmethod
  def apply(
    self, signal: npt.NDArray[np.complex64], sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Apply this impairment to the signal.

    Args:
      signal: Complex baseband signal.
      sample_rate: Sample rate in Hz.

    Returns:
      Impaired signal.
    """

  @property
  @abstractmethod
  def stage(self) -> ImpairmentStage:
    """The stage at which this impairment is applied."""

  @property
  @abstractmethod
  def name(self) -> str:
    """Human-readable name for this impairment."""


# ============================================================================
# Concrete Impairment Implementations
# ============================================================================


class AWGN(BaseModel, ChannelImpairment):
  """Additive White Gaussian Noise impairment.

  Adds thermal noise to the signal at a specified SNR.
  """

  snr_db: float
  seed: int | None = None

  model_config = {"frozen": True}

  def __init__(self, **data) -> None:
    super().__init__(**data)
    self._rng = np.random.default_rng(self.seed)

  def apply(
    self, signal: npt.NDArray[np.complex64], sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Add white Gaussian noise at specified SNR."""
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (self.snr_db / 10)
    noise_power = signal_power / snr_linear

    # Complex AWGN: noise power split equally between I and Q
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (
      self._rng.standard_normal(len(signal))
      + 1j * self._rng.standard_normal(len(signal))
    ).astype(np.complex64)

    return signal + noise

  @property
  def stage(self) -> ImpairmentStage:
    return ImpairmentStage.NOISE

  @property
  def name(self) -> str:
    return f"AWGN({self.snr_db}dB)"


class SSBFilter(BaseModel, ChannelImpairment):
  """SSB bandpass filter impairment.

  Models the typical 300-2700 Hz filter found in HF transceivers.
  """

  low_cutoff_hz: float = 300.0
  high_cutoff_hz: float = 2700.0
  filter_order: int = 5

  model_config = {"frozen": True}

  def __init__(self, **data) -> None:
    super().__init__(**data)
    self._sos_cache: dict[int, npt.NDArray] = {}

  def _get_filter_sos(self, sample_rate: int) -> npt.NDArray:
    """Get or create filter coefficients for given sample rate."""
    if sample_rate not in self._sos_cache:
      nyquist = 0.5 * sample_rate
      low = self.low_cutoff_hz / nyquist
      high = self.high_cutoff_hz / nyquist

      if low <= 0 or high >= 1 or low >= high:
        msg = (
          f"Invalid filter cutoffs: {self.low_cutoff_hz}-{self.high_cutoff_hz} Hz "
          f"for sample rate {sample_rate} Hz"
        )
        raise ValueError(msg)

      # Calculate required order for 30dB stopband attenuation
      # Transition width: 100 Hz
      transition = 100.0 / nyquist

      # Ensure transition doesn't go below DC or above Nyquist
      ws_low = max(0.001, low - transition)
      ws_high = min(0.999, high + transition)

      # Use cheb2ord to find minimum order for Chebyshev Type II
      # Cheby2 has flat passband (no ripple) and equiripple stopband
      # gpass=3dB (passband corner attenuation), gstop=30dB (stopband attenuation)
      filter_order, wn = scipy_signal.cheb2ord(
        [low, high], [ws_low, ws_high], gpass=3, gstop=30, fs=None
      )

      # Cap order to avoid extreme values
      filter_order = min(filter_order, 20)

      # Use Chebyshev Type II filter
      # rs is the stopband attenuation in dB
      self._sos_cache[sample_rate] = scipy_signal.cheby2(
        filter_order, rs=30, Wn=wn, btype="band", output="sos"
      )

    return self._sos_cache[sample_rate]

  def apply(
    self, signal: npt.NDArray[np.complex64], sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Apply SSB bandpass filter."""
    sos = self._get_filter_sos(sample_rate)

    # Filter I and Q independently
    filtered_real = scipy_signal.sosfilt(sos, signal.real)
    filtered_imag = scipy_signal.sosfilt(sos, signal.imag)

    return (filtered_real + 1j * filtered_imag).astype(np.complex64)

  @property
  def stage(self) -> ImpairmentStage:
    return ImpairmentStage.TRANSMITTER

  @property
  def name(self) -> str:
    return f"SSB({int(self.low_cutoff_hz)}-{int(self.high_cutoff_hz)}Hz)"


class FreqOffset(BaseModel, ChannelImpairment):
  """Carrier frequency offset impairment.

  Models LO mismatch between transmitter and receiver.
  """

  offset_hz: float

  model_config = {"frozen": True}

  def apply(
    self, signal: npt.NDArray[np.complex64], sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Apply carrier frequency offset."""
    if self.offset_hz == 0:
      return signal

    t = np.arange(len(signal), dtype=np.float64) / sample_rate
    offset = np.exp(2j * np.pi * self.offset_hz * t).astype(np.complex64)
    return signal * offset

  @property
  def stage(self) -> ImpairmentStage:
    return ImpairmentStage.PROPAGATION

  @property
  def name(self) -> str:
    return f"FreqOffset({self.offset_hz}Hz)"


class RayleighFading(BaseModel, ChannelImpairment):
  """Rayleigh fading impairment using Jakes model.

  Implements flat Rayleigh fading with Doppler spread using the
  sum-of-sinusoids (Jakes) method. Models non-line-of-sight (NLOS)
  mobile or ionospheric channels.

  References:
    - W.C. Jakes, "Microwave Mobile Communications", 1974
    - Y.R. Zheng & C. Xiao, "Improved models for the generation of multiple
      uncorrelated Rayleigh fading waveforms", IEEE Commun. Lett., 2002
  """

  doppler_hz: float = Field(default=0.0, ge=0.0)
  num_sinusoids: int = 16
  seed: int | None = None

  model_config = {"frozen": True}

  def __init__(self, **data) -> None:
    super().__init__(**data)
    self._rng = np.random.default_rng(self.seed)
    self._setup_jakes_oscillators()

  def _setup_jakes_oscillators(self) -> None:
    """Initialize Jakes model oscillators for fading generation."""
    if self.doppler_hz == 0:
      self._oscillators = None
      return

    n_sinusoids = self.num_sinusoids

    # Generate random phases (uniformly distributed)
    self._phases = self._rng.uniform(0, 2 * np.pi, n_sinusoids)

    # Generate Doppler frequencies using Jakes spectrum
    # f_n = f_d * cos(2π n / N) for n = 1, 2, ..., N
    n = np.arange(1, n_sinusoids + 1)
    self._doppler_freqs = self.doppler_hz * np.cos(2 * np.pi * n / n_sinusoids)

    # Amplitudes (normalized so E[|h|^2] = 1 for Rayleigh)
    # cos and sin have variance 1/2, so we need sqrt(2) factor
    self._amplitudes = np.ones(n_sinusoids) * np.sqrt(2.0 / n_sinusoids)

    self._oscillators = True

  def apply(
    self, signal: npt.NDArray[np.complex64], sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Apply Rayleigh fading using Jakes model."""
    if self.doppler_hz == 0 or self._oscillators is None:
      return signal

    t = np.arange(len(signal), dtype=np.float64) / sample_rate

    # Generate I and Q components separately
    h_i = np.zeros(len(signal), dtype=np.float64)
    h_q = np.zeros(len(signal), dtype=np.float64)

    for amp, freq, phase in zip(
      self._amplitudes, self._doppler_freqs, self._phases, strict=False
    ):
      h_i += amp * np.cos(2 * np.pi * freq * t + phase)
      h_q += amp * np.sin(2 * np.pi * freq * t + phase)

    h = (h_i + 1j * h_q).astype(np.complex128)

    return (signal * h).astype(np.complex64)

  @property
  def stage(self) -> ImpairmentStage:
    return ImpairmentStage.CHANNEL

  @property
  def name(self) -> str:
    return f"Rayleigh(fd={self.doppler_hz}Hz)"


class RicianFading(BaseModel, ChannelImpairment):
  """Rician fading impairment using Jakes model.

  Implements Rician fading with Doppler spread and a line-of-sight (LOS)
  component. The K-factor controls the ratio of LOS to scattered power.

  K-factor (dB):
    - 0 dB: Equal LOS and scattered power
    - 10 dB: Strong LOS (satellite, rural)
    - -∞ dB: No LOS (reduces to Rayleigh)
  """

  doppler_hz: float = Field(default=0.0, ge=0.0)
  k_factor_db: float
  num_sinusoids: int = 16
  seed: int | None = None

  model_config = {"frozen": True}

  def __init__(self, **data) -> None:
    super().__init__(**data)
    self._rng = np.random.default_rng(self.seed)
    self._setup_jakes_oscillators()

  def _setup_jakes_oscillators(self) -> None:
    """Initialize Jakes model oscillators for fading generation."""
    if self.doppler_hz == 0:
      self._oscillators = None
      return

    n_sinusoids = self.num_sinusoids

    # Generate random phases
    self._phases = self._rng.uniform(0, 2 * np.pi, n_sinusoids)

    # Doppler frequencies
    n = np.arange(1, n_sinusoids + 1)
    self._doppler_freqs = self.doppler_hz * np.cos(2 * np.pi * n / n_sinusoids)

    # Amplitudes for scattered component
    self._amplitudes = np.ones(n_sinusoids) * np.sqrt(2.0 / n_sinusoids)

    # LOS component scaling
    k_linear = 10 ** (self.k_factor_db / 10)
    self._los_amplitude = np.sqrt(k_linear / (k_linear + 1))
    self._scatter_scale = np.sqrt(1 / (k_linear + 1))
    self._los_phase = self._rng.uniform(0, 2 * np.pi)

    self._oscillators = True

  def apply(
    self, signal: npt.NDArray[np.complex64], sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Apply Rician fading using Jakes model."""
    if self.doppler_hz == 0 or self._oscillators is None:
      return signal

    t = np.arange(len(signal), dtype=np.float64) / sample_rate

    # Generate scattered component (Rayleigh)
    h_i = np.zeros(len(signal), dtype=np.float64)
    h_q = np.zeros(len(signal), dtype=np.float64)

    for amp, freq, phase in zip(
      self._amplitudes, self._doppler_freqs, self._phases, strict=False
    ):
      h_i += amp * np.cos(2 * np.pi * freq * t + phase)
      h_q += amp * np.sin(2 * np.pi * freq * t + phase)

    h_scatter = (h_i + 1j * h_q).astype(np.complex128)

    # Add LOS component
    h_los = self._los_amplitude * np.exp(1j * self._los_phase)
    h = h_los + self._scatter_scale * h_scatter

    return (signal * h).astype(np.complex64)

  @property
  def stage(self) -> ImpairmentStage:
    return ImpairmentStage.CHANNEL

  @property
  def name(self) -> str:
    return f"Rician(K={self.k_factor_db}dB,fd={self.doppler_hz}Hz)"


class TapProfile(BaseModel):
  """Configuration for a single channel tap in a Tapped Delay Line model.

  Attributes:
    delay_sec: Delay of this tap relative to the first tap in seconds.
    power_db: Average power of this tap relative to the strongest tap in dB.
              (Usually 0 dB for the first/strongest tap, negative for others)
    fading_model: Optional fading model (Rayleigh/Rician) to apply to this tap.
                  If None, the tap is static (no fading).
  """

  delay_sec: float = Field(ge=0.0)
  power_db: float = Field(le=0.0)
  fading_model: RayleighFading | RicianFading | None = None

  model_config = {"frozen": True}


class TappedDelayLine(BaseModel, ChannelImpairment):
  """Multipath channel simulator using Tapped Delay Line (TDL) model.

  Simulates frequency-selective fading by summing multiple delayed and
  independently faded copies of the signal. This models the physical reality
  of multipath propagation where signals arrive via different paths with
  different delays and Doppler shifts.

  The output signal is:
    y(t) = sum( h_i(t) * x(t - tau_i) )

  Where:
    - x(t) is the input signal
    - tau_i is the delay of the i-th tap
    - h_i(t) is the complex channel coefficient for the i-th tap (includes
      power scaling and fading)

  Note:
    This implementation uses integer sample delays (`int(round(delay * fs))`).
    For narrowband signals (low sample rates) or very small delays (nanoseconds),
    multiple taps may collapse into a single tap (flat fading). This is
    physically correct for narrowband simulations where the signal bandwidth
    cannot resolve the multipath delay spread.
  """

  taps: Sequence[TapProfile]
  normalize_power: bool = True

  model_config = {"frozen": True}

  def apply(
    self, signal: npt.NDArray[np.complex64], sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Apply Tapped Delay Line multipath model."""
    if not self.taps:
      return signal

    output_signal = np.zeros_like(signal)
    total_power_linear = 0.0

    for tap in self.taps:
      # 1. Calculate linear power scale
      power_linear = 10 ** (tap.power_db / 10)
      total_power_linear += power_linear
      amplitude_scale = np.sqrt(power_linear)

      # 2. Apply delay
      # Convert delay from seconds to samples
      delay_samples = tap.delay_sec * sample_rate

      # Use integer delay for simplicity and efficiency
      # (Fractional delay would require sinc interpolation, which is
      # computationally expensive and maybe overkill for this simulation
      # level, but can be added if needed)
      delay_int = round(delay_samples)

      if delay_int == 0:
        delayed_signal = signal
      elif delay_int >= len(signal):
        # Delay is longer than signal, contribution is zero
        delayed_signal = np.zeros_like(signal)
      else:
        # Shift signal
        delayed_signal = np.zeros_like(signal)
        delayed_signal[delay_int:] = signal[:-delay_int]

      # 3. Apply fading (if configured)
      if tap.fading_model:
        # Apply fading to the delayed signal
        # Note: We pass sample_rate to the fading model so it generates correct Doppler
        faded_signal = tap.fading_model.apply(delayed_signal, sample_rate)
      else:
        faded_signal = delayed_signal

      # 4. Accumulate
      output_signal += amplitude_scale * faded_signal

    # 5. Normalize total power if requested
    # This ensures the channel doesn't amplify or attenuate the total
    # signal energy on average
    if self.normalize_power and total_power_linear > 0:
      output_signal /= np.sqrt(total_power_linear)

    return output_signal

  @property
  def stage(self) -> ImpairmentStage:
    return ImpairmentStage.CHANNEL

  @property
  def name(self) -> str:
    return f"TDL({len(self.taps)} taps)"


class PhaseNoise(BaseModel, ChannelImpairment):
  """Oscillator phase noise impairment.

  Models phase noise as a Wiener process (random walk phase).
  The phase step variance is determined by the level_dbchz parameter,
  which represents the phase noise density at a specific offset.

  Note: This is a simplified model assuming 1/f^2 spectrum (white FM noise).
  """

  level_dbchz: float = Field(default=-80.0, le=0.0)
  offset_hz: float = Field(default=1000.0, gt=0.0)
  seed: int | None = None

  model_config = {"frozen": True}

  def apply(
    self, signal: npt.NDArray[np.complex64], sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Apply phase noise."""
    if self.level_dbchz <= -170:
      # Negligible noise
      return signal

    rng = np.random.default_rng(self.seed)

    # Calculate phase variance per sample
    # L(f) = 10*log10(S_phi(f)/2)
    # S_phi(f) = C / f^2 for Wiener process
    # C = 2 * 10^(L/10) * f_offset^2
    # Variance per second = 2 * pi^2 * C
    # Variance per sample = Variance per second / fs

    l_linear = 10 ** (self.level_dbchz / 10)
    c = 2 * l_linear * (self.offset_hz**2)
    var_per_sample = (2 * np.pi**2 * c) / sample_rate
    std_per_sample = np.sqrt(var_per_sample)

    # Generate random phase steps
    steps = rng.normal(0, std_per_sample, len(signal))

    # Integrate to get phase (random walk)
    phase_noise = np.cumsum(steps)

    # Apply to signal
    return (signal * np.exp(1j * phase_noise)).astype(np.complex64)

  @property
  def stage(self) -> ImpairmentStage:
    return ImpairmentStage.TRANSMITTER

  @property
  def name(self) -> str:
    return f"PhaseNoise({self.level_dbchz}dBc/Hz)"


class Nonlinearity(BaseModel, ChannelImpairment):
  """Power amplifier nonlinearity using Rapp model.

  Implements the Rapp soft-clipping model for PA saturation. This models
  the AM/AM (amplitude-to-amplitude) characteristic of a solid-state PA.

  The Rapp model:
    output = input / (1 + |input/A_sat|^(2p))^(1/2p)

  Where:
    - A_sat: Saturation amplitude (set by input_backoff_db)
    - p: Smoothness parameter (higher = harder clipping)

  References:
    - C. Rapp, "Effects of HPA-Nonlinearity on a 4-DPSK/OFDM-Signal for a
      Digital Sound Broadcasting System", ESA, 1991
  """

  input_backoff_db: float = Field(default=0.0, ge=0.0)
  smoothness: float = Field(default=2.0, gt=0.0)

  model_config = {"frozen": True}

  def apply(
    self, signal: npt.NDArray[np.complex64], sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Apply Rapp model nonlinearity."""
    if self.input_backoff_db == 0:
      # No saturation
      return signal

    # Calculate saturation amplitude from input backoff
    # IBO (dB) = 20*log10(A_sat / A_in)
    # For unit power signal, A_in ≈ 1, so A_sat = 10^(IBO/20)
    a_sat = 10 ** (self.input_backoff_db / 20)

    # Extract amplitude and phase
    amplitude = np.abs(signal)
    phase = np.angle(signal)

    # Apply Rapp model to amplitude
    # output_amp = input_amp / (1 + (input_amp/A_sat)^(2p))^(1/2p)
    p = self.smoothness
    ratio = amplitude / a_sat
    compression_factor = 1.0 / (1.0 + ratio ** (2 * p)) ** (1.0 / (2 * p))
    output_amplitude = amplitude * compression_factor

    # Reconstruct signal (phase unchanged - AM/AM only, no AM/PM)
    return (output_amplitude * np.exp(1j * phase)).astype(np.complex64)

  @property
  def stage(self) -> ImpairmentStage:
    return ImpairmentStage.TRANSMITTER

  @property
  def name(self) -> str:
    return f"Nonlinearity(IBO={self.input_backoff_db}dB,p={self.smoothness})"


class Transmitter(ABC):
  """Abstract base class for radio transmitters.

  A transmitter converts audio signals into modulated RF (baseband I/Q) signals
  suitable for transmission over a radio channel.
  """

  @abstractmethod
  def modulate(
    self, audio: npt.NDArray[np.float32], input_sample_rate: int
  ) -> npt.NDArray[np.complex64]:
    """Convert audio to modulated RF signal.

    Args:
      audio: Input audio samples (mono, float32, normalized to [-1, 1]).
      input_sample_rate: Audio sample rate in Hz.

    Returns:
      Complex baseband signal (I/Q samples) at the transmitter's output sample rate.
    """

  @property
  @abstractmethod
  def name(self) -> str:
    """Human-readable transmitter name for reporting."""

  @property
  @abstractmethod
  def bandwidth_hz(self) -> float:
    """Signal bandwidth in Hz (occupied bandwidth)."""

  @property
  @abstractmethod
  def output_sample_rate(self) -> int:
    """Sample rate of modulated output in Hz."""


class Receiver(ABC):
  """Abstract base class for radio receivers.

  A receiver demodulates received RF (baseband I/Q) signals back into audio.
  """

  @abstractmethod
  def demodulate(
    self, signal: npt.NDArray[np.complex64], input_sample_rate: int
  ) -> npt.NDArray[np.float32]:
    """Convert received RF signal back to audio.

    Args:
      signal: Complex baseband signal (I/Q samples).
      input_sample_rate: Signal sample rate in Hz.

    Returns:
      Demodulated audio (mono, float32, normalized to [-1, 1]).
    """

  @property
  @abstractmethod
  def name(self) -> str:
    """Human-readable receiver name for reporting."""

  @property
  @abstractmethod
  def output_sample_rate(self) -> int:
    """Sample rate of demodulated audio in Hz."""


class ChannelSimulator:
  """Simulates realistic radio channel impairments using composable effects.

  This class applies channel impairments in a physically realistic order:
  1. TRANSMITTER: SSB filter, PA nonlinearity, TX phase noise
  2. CHANNEL: Fading, multipath
  3. PROPAGATION: Frequency offset, Doppler
  4. NOISE: AWGN

  Impairments are automatically sorted by stage regardless of input order.
  """

  def __init__(
    self, impairments: Sequence[ChannelImpairment], sample_rate: int
  ) -> None:
    """Initialize channel simulator with impairments.

    Args:
      impairments: List of impairment objects to apply.
      sample_rate: Sample rate of the signal in Hz.
    """
    self.sample_rate = sample_rate

    # Sort impairments by stage
    original_names = [imp.name for imp in impairments]
    self.impairments = sorted(impairments, key=lambda x: x.stage)
    sorted_names = [imp.name for imp in self.impairments]

    # Warn if order changed
    if original_names != sorted_names:
      logger.warning(
        f"Impairments reordered to canonical sequence:\n"
        f"  User order: {original_names}\n"
        f"  Canonical:  {sorted_names}"
      )

  def apply(self, signal: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    """Apply all impairments to signal in canonical order.

    Args:
      signal: Complex baseband signal.

    Returns:
      Impaired signal with same shape as input.
    """
    result = signal.copy()

    for impairment in self.impairments:
      result = impairment.apply(result, self.sample_rate)

    return result

  @property
  def name(self) -> str:
    """Human-readable description of channel condition."""
    return "_".join(imp.name for imp in self.impairments)

  @property
  def snr_db(self) -> float | None:
    """Get SNR in dB if AWGN impairment is present."""
    for imp in self.impairments:
      if isinstance(imp, AWGN):
        return imp.snr_db
    return None


class TransmissionSystem:
  """Complete radio transmission system: TX -> Channel -> RX.

  This class orchestrates the complete signal chain from audio input through
  modulation, channel impairments, and demodulation back to audio.
  """

  def __init__(
    self, transmitter: Transmitter, receiver: Receiver, channel: ChannelSimulator
  ) -> None:
    """Initialize transmission system.

    Args:
      transmitter: Transmitter instance.
      receiver: Receiver instance.
      channel: Channel simulator instance.
    """
    self.transmitter = transmitter
    self.receiver = receiver
    self.channel = channel

  def process(
    self, audio: npt.NDArray[np.float32], audio_sample_rate: int
  ) -> npt.NDArray[np.float32]:
    """Process audio through complete TX -> Channel -> RX chain.

    Args:
      audio: Input audio samples (mono, float32, normalized to [-1, 1]).
      audio_sample_rate: Audio sample rate in Hz.

    Returns:
      Received audio after channel impairments, at receiver's output sample rate.
    """
    # Transmit
    modulated = self.transmitter.modulate(audio, audio_sample_rate)

    # Channel
    received_signal = self.channel.apply(modulated)

    # Receive
    return self.receiver.demodulate(
      received_signal, self.transmitter.output_sample_rate
    )

  @property
  def name(self) -> str:
    """System name for reporting (TX_RX format)."""
    return f"{self.transmitter.name}_{self.receiver.name}"
