"""Tests for transmission system components."""

import numpy as np
import numpy.typing as npt
import scipy

from shannon_bench.simulator.impairements import (
  AWGN,
  ChannelSimulator,
  FreqOffset,
  SSBFilter,
)
from shannon_bench.simulator.transmission_system import (
  Receiver,
  SourceDecoder,
  SourceEncoder,
  TransmissionSystem,
  Transmitter,
)


class MockSourceEncoder(SourceEncoder):
  """MOCK: Passes audio through, resampling it to the output_sample_rate."""

  def __init__(self, output_sample_rate: int = 48000) -> None:
    self._output_sample_rate = output_sample_rate

  @property
  def output_sample_rate(self) -> float:
    # The output rate is the sample rate in Hz
    return float(self._output_sample_rate)

  def encode(
    self, audio: npt.NDArray[np.float32], input_sample_rate: int
  ) -> npt.NDArray[np.float32]:
    """Resamples the input audio to the output sample rate."""
    if input_sample_rate == self._output_sample_rate:
      return audio

    # Use polyphase resampling for simulation accuracy
    gcd = np.gcd(input_sample_rate, self._output_sample_rate)
    up = self._output_sample_rate // gcd
    down = input_sample_rate // gcd

    # Ensure audio is float64 for intermediate processing in resample_poly
    audio_f64 = audio.astype(np.float64)
    resampled_audio = scipy.signal.resample_poly(audio_f64, up, down)

    # Return as float32, which is the expected type for processed audio
    return resampled_audio.astype(np.float32)


class MockSourceDecoder(SourceDecoder):
  """MOCK: Passes audio through, resampling it to the output_sample_rate."""

  def __init__(self, output_sample_rate: int = 48000) -> None:
    self._output_sample_rate = output_sample_rate

  @property
  def output_sample_rate(self) -> float:
    # The output rate is the sample rate in Hz
    return float(self._output_sample_rate)

  def decode(
    self, demod_output: npt.NDArray[np.float32], input_rate: float
  ) -> npt.NDArray[np.float32]:
    """Resamples the input audio to the output sample rate."""
    if input_rate == self._output_sample_rate:
      return demod_output

    # Use polyphase resampling for simulation accuracy
    gcd = np.gcd(int(input_rate), self._output_sample_rate)
    up = self._output_sample_rate // gcd
    down = int(input_rate) // gcd

    # Ensure audio is float64 for intermediate processing in resample_poly
    demod_output_f64 = demod_output.astype(np.float64)
    resampled_audio = scipy.signal.resample_poly(demod_output_f64, up, down)

    # Return as float32, which is the expected type for processed audio
    return resampled_audio.astype(np.float32)


class MockTransmitter(Transmitter):
  """Mock transmitter for testing."""

  def __init__(
    self, tx_name: str = "MockTX", sample_rate: int = 8000, bw: float = 3000.0
  ) -> None:
    self._name = tx_name
    self._sample_rate = sample_rate
    self._bandwidth = bw

  def modulate(
    self,
    source_output: npt.NDArray[np.float32] | npt.NDArray[np.uint8],
    source_output_rate: float,
  ) -> np.ndarray:
    """Simply return audio as complex signal (identity modulation)."""
    return source_output.astype(np.complex64)

  @property
  def name(self) -> str:
    return self._name

  @property
  def bandwidth_hz(self) -> float:
    return self._bandwidth

  @property
  def output_sample_rate(self) -> int:
    return self._sample_rate


class MockReceiver(Receiver):
  """Mock receiver for testing."""

  def __init__(self, rx_name: str = "MockRX", sample_rate: int = 8000) -> None:
    self._name = rx_name
    self._sample_rate = sample_rate

  def demodulate(self, signal: np.ndarray, input_sample_rate: int) -> np.ndarray:
    """Simply return real part of signal (identity demodulation)."""
    return signal.real.astype(np.float32)

  @property
  def name(self) -> str:
    return self._name

  @property
  def output_sample_rate(self) -> int:
    return self._sample_rate


class TestChannelSimulatorProperties:
  """Tests for ChannelSimulator properties."""

  def test_name_property(self) -> None:
    """Test that name property returns joined impairment names."""
    impairments = [
      SSBFilter(low_cutoff_hz=300, high_cutoff_hz=2700),
      FreqOffset(offset_hz=50),
      AWGN(snr_db=20),
    ]
    sim = ChannelSimulator(impairments=impairments, sample_rate=8000)

    name = sim.name
    assert "SSB" in name
    assert "FreqOffset" in name
    assert "AWGN" in name
    assert "_" in name  # Names should be joined with underscore

  def test_snr_db_with_awgn(self) -> None:
    """Test snr_db property when AWGN is present."""
    target_snr = 15.0
    impairments = [
      SSBFilter(),
      AWGN(snr_db=target_snr),
    ]
    sim = ChannelSimulator(impairments=impairments, sample_rate=8000)

    assert sim.snr_db == target_snr

  def test_snr_db_without_awgn(self) -> None:
    """Test snr_db property returns None when no AWGN."""
    impairments = [
      SSBFilter(),
      FreqOffset(offset_hz=100),
    ]
    sim = ChannelSimulator(impairments=impairments, sample_rate=8000)

    assert sim.snr_db is None


class TestTransmissionSystem:
  """Tests for TransmissionSystem."""

  def test_initialization(self) -> None:
    """Test TransmissionSystem initialization."""
    se = MockSourceEncoder()
    sd = MockSourceDecoder()
    tx = MockTransmitter(tx_name="TestTX")
    rx = MockReceiver(rx_name="TestRX")
    channel = ChannelSimulator(impairments=[], sample_rate=8000)

    system = TransmissionSystem(
      source_encoder=se, source_decoder=sd, transmitter=tx, receiver=rx, channel=channel
    )

    assert system.source_encoder is se
    assert system.source_decoder is sd
    assert system.transmitter is tx
    assert system.receiver is rx
    assert system.channel is channel

  def test_process(self) -> None:
    """Test complete signal processing chain."""
    # The rate the raw audio comes in at
    AUDIO_RATE = 8000
    # The high sample rate used for modulation, channel simulation, and demodulation
    MOD_RATE = 48000

    se = MockSourceEncoder(output_sample_rate=AUDIO_RATE)
    tx = MockTransmitter(sample_rate=MOD_RATE)
    rx = MockReceiver(sample_rate=MOD_RATE)
    sd = MockSourceDecoder(output_sample_rate=AUDIO_RATE)
    impairments = [AWGN(snr_db=20, seed=42)]
    channel = ChannelSimulator(impairments=impairments, sample_rate=MOD_RATE)

    system = TransmissionSystem(
      source_encoder=se, source_decoder=sd, transmitter=tx, receiver=rx, channel=channel
    )

    # Create test audio
    audio = np.random.randn(1000).astype(np.float32)
    audio_sample_rate = 8000

    # Process through system
    output = system.process(audio, audio_sample_rate)

    # Output should be same length as input (for our mock implementation)
    assert len(output) == len(audio)
    assert output.dtype == np.float32

    # Output should be different from input (noise added)
    assert not np.allclose(output, audio)

  def test_name_property(self) -> None:
    """Test system name property."""
    tx = MockTransmitter(tx_name="SSB_TX")
    rx = MockReceiver(rx_name="SSB_RX")
    se = MockSourceEncoder()
    sd = MockSourceDecoder()
    channel = ChannelSimulator(impairments=[], sample_rate=8000)

    system = TransmissionSystem(
      source_encoder=se, source_decoder=sd, transmitter=tx, receiver=rx, channel=channel
    )

    assert system.name == "SSB_TX_SSB_RX"
