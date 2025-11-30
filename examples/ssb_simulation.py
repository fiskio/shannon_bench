#!/usr/bin/env python3
"""SSB Transmission Simulation Script.

This script simulates a complete SSB transmission chain:
Audio Input -> SSB Transmitter -> Channel Simulator -> SSB Receiver -> Audio Output

It allows testing different channel conditions and listening to the effects
of impairments like fading, noise, and frequency offsets.
"""

import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import scipy.io.wavfile
import sounddevice as sd
import typer

from shannon_bench.setup_logging import setup_logging
from shannon_bench.simulator.channels import (
  ChannelPreset,
  eme,
  hf,
  ideal,
  satellite,
  vhf,
)
from shannon_bench.simulator.impairements import ChannelSimulator
from shannon_bench.simulator.ssb_system import (
  AnalogSourceDecoder,
  AnalogSourceEncoder,
  SSBReceiver,
  SSBTransmitter,
)
from shannon_bench.simulator.transmission_system import TransmissionSystem

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


class SidebandMode(StrEnum):
  """Sideband mode enum."""

  USB = "USB"
  LSB = "LSB"


def load_audio(file_path: Path) -> tuple[int, np.ndarray]:
  """Load audio file and convert to float32 [-1, 1]."""
  try:
    sample_rate, data = scipy.io.wavfile.read(file_path)
  except (ValueError, OSError):
    logger.exception(f"Error reading audio file {file_path}")
    sys.exit(1)

  # Convert to float32 normalized
  if data.dtype == np.int16:
    data = data.astype(np.float32) / 32768.0
  elif data.dtype == np.int32:
    data = data.astype(np.float32) / 2147483648.0
  elif data.dtype == np.uint8:
    data = (data.astype(np.float32) - 128.0) / 128.0
  elif data.dtype == np.float32:
    pass  # Already float32
  else:
    logger.error(f"Unsupported audio format: {data.dtype}")
    sys.exit(1)

  # Convert to mono if stereo
  if len(data.shape) > 1:
    logger.info("Converting stereo to mono...")
    data = np.mean(data, axis=1)

  return sample_rate, data


def save_audio(file_path: Path, sample_rate: int, data: np.ndarray) -> None:
  """Save audio file as int16 wav."""
  # Clip to [-1, 1]
  data = np.clip(data, -1.0, 1.0)
  # Convert to int16
  data_int16 = (data * 32767).astype(np.int16)
  scipy.io.wavfile.write(file_path, sample_rate, data_int16)
  logger.info(f"Saved output to {file_path}")


def play_audio(sample_rate: int, data: np.ndarray) -> None:
  """Play audio using sounddevice."""
  # Clip to [-1, 1]
  data = np.clip(data, -1.0, 1.0)
  logger.info("Playing audio... (press Ctrl+C to stop)")
  try:
    sd.play(data, sample_rate, blocking=True)
    logger.info("Playback complete!")
  except KeyboardInterrupt:
    sd.stop()
    logger.info("Playback stopped.")


def get_channel_preset(preset_name: str) -> ChannelPreset:
  """Get channel preset by name."""
  presets = {
    "ideal": ideal.PERFECT,
    "hf_excellent": hf.ITU_R_EXCELLENT,
    "hf_good": hf.ITU_R_GOOD,
    "hf_moderate": hf.ITU_R_MODERATE,
    "hf_poor": hf.ITU_R_POOR,
    "hf_nvis": hf.NVIS,
    "vhf_fixed": vhf.FIXED_LOS,
    "vhf_urban": vhf.MOBILE_URBAN,
    "vhf_rural": vhf.MOBILE_RURAL,
    "sat_leo": satellite.LEO_CLEAR,
    "sat_geo": satellite.GEO,
    "eme_smooth": eme.SMOOTH_MOON,
  }

  if preset_name not in presets:
    logger.error(f"Unknown preset: {preset_name}")
    logger.error(f"Available presets: {', '.join(presets.keys())}")
    sys.exit(1)

  return presets[preset_name]


def main(
  input_file: Annotated[
    Path, typer.Argument(help="Input WAV file path", exists=True, readable=True)
  ],
  output: Annotated[
    Path | None,
    typer.Option(
      "--output",
      "-o",
      help="Output WAV file path (if not specified, plays audio instead)",
    ),
  ] = None,
  preset: Annotated[
    str,
    typer.Option(
      "--preset",
      "-p",
      help="Channel preset (e.g., hf_poor, vhf_urban).",
    ),
  ] = "hf_poor",
  mode: Annotated[
    SidebandMode,
    typer.Option("--mode", "-m", help="SSB mode (USB or LSB)."),
  ] = SidebandMode.USB,
  bandwidth: Annotated[
    float,
    typer.Option("--bandwidth", "-b", help="Signal bandwidth in Hz."),
  ] = 3000.0,
) -> None:
  """Simulate SSB transmission over an impaired channel."""
  # 1. Load Audio
  logger.info(f"Loading {input_file}...")
  sample_rate, audio = load_audio(input_file)
  logger.info(f"Loaded {len(audio) / sample_rate:.2f}s audio at {sample_rate}Hz")

  # 2. Setup System
  logger.info("Setting up simulation chain...")

  # Source encoder/decoder (pass-through for analog systems)
  source_encoder = AnalogSourceEncoder(output_sample_rate=sample_rate)
  source_decoder = AnalogSourceDecoder(output_sample_rate=sample_rate)

  # Transmitter
  tx = SSBTransmitter(
    bandwidth_hz=bandwidth,
    output_sample_rate=48000,  # Standard internal rate
    mode=cast("Literal['USB', 'LSB']", mode.value),
  )

  # Receiver
  rx = SSBReceiver(
    output_sample_rate=sample_rate,  # Match input rate for easy comparison
    mode=cast("Literal['USB', 'LSB']", mode.value),
  )

  # Channel
  channel_preset = get_channel_preset(preset)
  logger.info(f"Using channel preset: {channel_preset.name}")
  logger.info(f"  {channel_preset.description}")

  channel = ChannelSimulator(
    impairments=channel_preset.impairments, sample_rate=tx.output_sample_rate
  )

  system = TransmissionSystem(
    source_encoder=source_encoder,
    source_decoder=source_decoder,
    transmitter=tx,
    receiver=rx,
    channel=channel,
  )

  # 3. Run Simulation
  logger.info("Running simulation...")
  output_audio = system.process(audio, sample_rate)

  # 4. Output (save or play)
  if output is not None:
    save_audio(output, sample_rate, output_audio)
    logger.info("Simulation complete!")
    logger.info("To compare, play the original and output files:")
    logger.info(f"  Original: {input_file}")
    logger.info(f"  Output:   {output}")
  else:
    logger.info("Simulation complete!")
    play_audio(sample_rate, output_audio)


if __name__ == "__main__":
  typer.run(main)
