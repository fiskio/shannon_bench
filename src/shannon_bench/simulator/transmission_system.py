"""Radio transmission system components.

This module defines the abstract base classes for transmitters and receivers,
as well as the TransmissionSystem class that orchestrates the complete
TX -> Channel -> RX signal chain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import numpy as np
  import numpy.typing as npt

  from shannon_bench.simulator.impairements import ChannelSimulator


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
