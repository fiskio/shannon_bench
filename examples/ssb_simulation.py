#!/usr/bin/env python3
"""SSB Transmission Simulation Script.

This script simulates a complete SSB transmission chain:
Audio Input -> SSB Transmitter -> Channel Simulator -> SSB Receiver -> Audio Output

It allows testing different channel conditions and listening to the effects
of impairments like fading, noise, and frequency offsets.
"""

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import scipy.io.wavfile
import typer

from shannon_bench.simulator.channels import ChannelPreset, eme, hf, satellite, vhf
from shannon_bench.simulator.impairements import ChannelSimulator
from shannon_bench.simulator.ssb_system import SSBReceiver, SSBTransmitter
from shannon_bench.simulator.transmission_system import TransmissionSystem


class SidebandMode(StrEnum):
  """Sideband mode enum."""

  USB = "USB"
  LSB = "LSB"


def load_audio(file_path: Path) -> tuple[int, np.ndarray]:
  """Load audio file and convert to float32 [-1, 1]."""
  try:
    sample_rate, data = scipy.io.wavfile.read(file_path)
  except (ValueError, OSError) as e:
    print(f"Error reading audio file {file_path}: {e}")  # noqa: T201
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
    print(f"Unsupported audio format: {data.dtype}")  # noqa: T201
    sys.exit(1)

  # Convert to mono if stereo
  if len(data.shape) > 1:
    print("Converting stereo to mono...")  # noqa: T201
    data = np.mean(data, axis=1)

  return sample_rate, data


def save_audio(file_path: Path, sample_rate: int, data: np.ndarray) -> None:
  """Save audio file as int16 wav."""
  # Clip to [-1, 1]
  data = np.clip(data, -1.0, 1.0)
  # Convert to int16
  data_int16 = (data * 32767).astype(np.int16)
  scipy.io.wavfile.write(file_path, sample_rate, data_int16)
  print(f"Saved output to {file_path}")  # noqa: T201


def get_channel_preset(preset_name: str) -> ChannelPreset:
  """Get channel preset by name."""
  presets = {
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
    print(f"Unknown preset: {preset_name}")  # noqa: T201
    print(f"Available presets: {', '.join(presets.keys())}")  # noqa: T201
    sys.exit(1)

  return presets[preset_name]


def main(
  input_file: Annotated[
    Path, typer.Argument(help="Input WAV file path", exists=True, readable=True)
  ],
  output: Annotated[
    Path, typer.Option("--output", "-o", help="Output WAV file path")
  ] = Path("output.wav"),
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
  print(f"Loading {input_file}...")  # noqa: T201
  sample_rate, audio = load_audio(input_file)
  print(f"Loaded {len(audio) / sample_rate:.2f}s audio at {sample_rate}Hz")  # noqa: T201

  # 2. Setup System
  print("Setting up simulation chain...")  # noqa: T201

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
  print(f"Using channel preset: {channel_preset.name}")  # noqa: T201
  print(f"  {channel_preset.description}")  # noqa: T201

  channel = ChannelSimulator(
    impairments=channel_preset.impairments, sample_rate=tx.output_sample_rate
  )

  system = TransmissionSystem(tx, rx, channel)

  # 3. Run Simulation
  print("Running simulation...")  # noqa: T201
  output_audio = system.process(audio, sample_rate)

  # 4. Save Output
  save_audio(output, sample_rate, output_audio)

  print("\nSimulation complete!")  # noqa: T201
  print("To compare, play the original and output files:")  # noqa: T201
  print(f"  Original: {input_file}")  # noqa: T201
  print(f"  Output:   {output}")  # noqa: T201


if __name__ == "__main__":
  typer.run(main)
