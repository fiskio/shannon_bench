"""Radio transmission system components.

This module defines the abstract base classes for transmitters and receivers,
as well as the TransmissionSystem class that orchestrates the complete
TX -> Channel -> RX signal chain.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import numpy as np
  import numpy.typing as npt

  from shannon_bench.simulator.impairements import ChannelSimulator


class SourceEncoder(ABC):
  """Abstract base class for source coding (audio to digital symbols/bits)."""

  @abstractmethod
  def encode(
    self, audio: npt.NDArray[np.float32], input_sample_rate: int
  ) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32]:
    """Convert audio to a representation suitable for modulation.

    For digital systems, this would be a stream of bits (np.uint8).
    For analog systems (like SSB/FM), this might be filtered/processed audio
    (np.float32).

    Args:
      audio: Input audio samples (mono, float32).
      input_sample_rate: Audio sample rate in Hz.

    Returns:
      Encoded output (bits, symbols, or processed audio).
    """

  @property
  @abstractmethod
  def output_sample_rate(self) -> int:
    """The output rate of the encoded signal (e.g., bits per second or sample rate)."""


class SourceDecoder(ABC):
  """Abstract base class for source decoding (digital symbols/bits back to audio)."""

  @abstractmethod
  def decode(
    self, demod_output: npt.NDArray[np.float32], input_rate: float
  ) -> npt.NDArray[np.float32]:
    """Convert demodulated output back to audio.

    Args:
      demod_output: The output from the Receiver's demodulation stage
        (e.g., recovered symbols or processed audio).
      input_rate: The rate of the input signal (e.g., symbols/sec or sample rate).

    Returns:
      Reconstructed audio (mono, float32, normalized to [-1, 1]).
    """

  @property
  @abstractmethod
  def output_sample_rate(self) -> int:
    """Sample rate of the final output audio in Hz."""


class TransmissionSystem:
  """Complete radio transmission system: TX -> Channel -> RX.

  This class orchestrates the complete signal chain from audio input through
  modulation, channel impairments, and demodulation back to audio.
  """

  def __init__(
    self,
    source_encoder: SourceEncoder,
    source_decoder: SourceDecoder,
    transmitter: Transmitter,
    receiver: Receiver,
    channel: ChannelSimulator,
  ) -> None:
    """Initialize transmission system.

    Args:
      source_encoder: Source encoder instance.
      source_decoder: Source decoder instance.
      transmitter: Transmitter instance.
      receiver: Receiver instance.
      channel: Channel simulator instance.
    """
    self.source_encoder = source_encoder
    self.source_decoder = source_decoder
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
    # 1. Source Coding (TX)
    encoded_output = self.source_encoder.encode(audio, audio_sample_rate)

    # 2. Modulation (TX)
    modulated = self.transmitter.modulate(
      encoded_output, self.source_encoder.output_sample_rate
    )

    # 3. Channel Impairments
    received_signal = self.channel.apply(modulated)

    # 4. Demodulation (RX)
    # The output rate of the demodulator is the same as the input rate
    # expected by the SourceDecoder, which is typically the rate
    # coming out of the SourceEncoder/Transmitter.
    demodulated_output = self.receiver.demodulate(
      received_signal,
      self.transmitter.output_sample_rate,  # TX output rate is RX input rate
    )

    # 5. Source Decoding (RX)
    return self.source_decoder.decode(
      demodulated_output,
      self.source_encoder.output_sample_rate,  # Decoder uses the Coder's expected rate
    )

  @property
  def name(self) -> str:
    """System name for reporting (TX_RX format)."""
    return f"{self.transmitter.name}_{self.receiver.name}"


class Receiver(ABC):
  """Abstract base class for radio receivers.

  A receiver demodulates received RF (baseband I/Q) signals back into audio.
  """

  @abstractmethod
  def demodulate(
    self, signal: npt.NDArray[np.complex64], input_sample_rate: int
  ) -> npt.NDArray[np.float32]:
    """Convert received RF signal to demodulated output (symbols or processed audio).

    Args:
      signal: Complex baseband signal (I/Q samples).
      input_sample_rate: Signal sample rate in Hz.

    Returns:
      Demodulated output (e.g., recovered symbols or processed audio).
    """

  @property
  @abstractmethod
  def name(self) -> str:
    """Human-readable receiver name for reporting."""

  @property
  @abstractmethod
  def output_sample_rate(self) -> int:
    """Sample rate of demodulated audio in Hz."""


class Transmitter(ABC):
  """Abstract base class for radio transmitters.

  A transmitter converts the SourceEncoder output (audio or symbols) into
  modulated RF (baseband I/Q) signals.
  """

  # The modulate method is simplified to take the source coder output
  @abstractmethod
  def modulate(
    self,
    source_output: npt.NDArray[np.float32] | npt.NDArray[np.uint8],
    source_output_rate: float,
  ) -> npt.NDArray[np.complex64]:
    """Convert the source coder output (audio or symbols) to modulated RF signal.

    Args:
      source_output: The output from the SourceEncoder
        (e.g., processed audio or bit stream).
      source_output_rate: The output rate from the SourceEncoder (Hz or bits/s).

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
