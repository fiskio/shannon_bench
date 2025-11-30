"""Standard library of pre-configured channel conditions.

This module provides factory functions and constants for common radio channel
scenarios, organized by propagation environment (HF, VHF/UHF, Satellite, EME).
Each preset returns a list of ChannelImpairment objects that can be used with
ChannelSimulator.

Typical Usage:
  ```python
  from shannon_bench.simulator.channels import hf
  from shannon_bench.simulator.transmission_system import ChannelSimulator

  # Use factory function with custom parameters
  impairments = hf.poor(snr_db=8.0, doppler_hz=1.5)
  channel = ChannelSimulator(impairments, sample_rate=8000)

  # Or use pre-configured preset
  channel = ChannelSimulator(hf.ITU_R_POOR, sample_rate=8000)
  ```

References:
  - ITU-R Rec. F.1487: "Testing of HF modems with bandwidths of up to
    about 12 kHz using ionospheric channel simulators"
  - 3GPP TS 36.101: "User Equipment (UE) radio transmission and reception"
  - CCIR Report 252-2: "Factors affecting the choice of frequencies for
    circuits using ionospheric propagation"
"""

from collections.abc import Sequence

from pydantic import BaseModel

from shannon_bench.simulator.transmission_system import (
  AWGN,
  ChannelImpairment,
  FreqOffset,
  Nonlinearity,
  PhaseNoise,
  RayleighFading,
  RicianFading,
  SSBFilter,
  TappedDelayLine,
  TapProfile,
)


class ChannelPreset(BaseModel):
  """A named channel configuration with metadata.

  Attributes:
    name: Human-readable preset name.
    description: Detailed description of channel characteristics.
    impairments: List of impairments to apply.
    typical_sample_rate: Recommended sample rate in Hz for this channel.
  """

  name: str
  description: str
  impairments: Sequence[ChannelImpairment]
  typical_sample_rate: int

  model_config = {"frozen": True, "arbitrary_types_allowed": True}


# =============================================================================
# HF (High Frequency) Channels - Ionospheric Propagation
# =============================================================================


class hf:  # noqa: N801
  """HF (3-30 MHz) channel presets for ionospheric propagation.

  HF channels are characterized by:
  - Ionospheric reflection (skywave propagation)
  - Multipath fading with Doppler spread (0.1-5 Hz typical)
  - Frequency-selective fading
  - SSB voice bandwidth: 300-2700 Hz
  - Typical sample rate: 8 kHz

  References:
    - ITU-R Rec. F.1487 (HF channel models)
    - CCIR Report 252-2 (HF propagation factors)
  """

  TYPICAL_SAMPLE_RATE: int = 8000  # Hz

  # Pre-configured presets (assigned below)
  ITU_R_EXCELLENT: ChannelPreset
  ITU_R_GOOD: ChannelPreset
  ITU_R_MODERATE: ChannelPreset
  ITU_R_POOR: ChannelPreset
  NVIS: ChannelPreset

  @staticmethod
  def excellent(
    snr_db: float = 30.0, freq_offset_hz: float = 1.0
  ) -> list[ChannelImpairment]:
    """Excellent HF conditions: minimal fading, very high SNR.

    Typical for:
    - Short-range skywave (< 500 km)
    - Quiet ionospheric conditions
    - Strong signal paths

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 30 dB).
      freq_offset_hz: Carrier frequency offset in Hz (default: 1 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      SSBFilter(low_cutoff_hz=300.0, high_cutoff_hz=2700.0),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def good(
    snr_db: float = 20.0,
    doppler_hz: float = 0.5,
    freq_offset_hz: float = 2.0,
  ) -> list[ChannelImpairment]:
    """Good HF conditions: light fading, high SNR.

    Typical for:
    - Stable mid-range paths (500-2000 km)
    - Daytime propagation
    - Low geomagnetic activity

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 20 dB).
      doppler_hz: Maximum Doppler shift in Hz (default: 0.5 Hz).
      freq_offset_hz: Carrier frequency offset in Hz (default: 2 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      SSBFilter(low_cutoff_hz=300.0, high_cutoff_hz=2700.0),
      # ITU-R F.1487 "Good" condition: 2 taps, 0.5ms delay spread
      TappedDelayLine(
        taps=[
          TapProfile(
            delay_sec=0.0,
            power_db=0.0,
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
          TapProfile(
            delay_sec=0.0005,  # 0.5 ms
            power_db=0.0,  # Equal power
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
        ]
      ),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def moderate(
    snr_db: float = 12.0,
    doppler_hz: float = 1.0,
    freq_offset_hz: float = 5.0,
  ) -> list[ChannelImpairment]:
    """Moderate HF conditions: moderate fading and noise.

    Typical for:
    - Long-range paths (2000-4000 km)
    - Nighttime propagation
    - Moderate geomagnetic activity

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 12 dB).
      doppler_hz: Maximum Doppler shift in Hz (default: 1 Hz).
      freq_offset_hz: Carrier frequency offset in Hz (default: 5 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      SSBFilter(low_cutoff_hz=300.0, high_cutoff_hz=2700.0),
      # ITU-R F.1487 "Moderate" condition: 2 taps, 1ms delay spread
      TappedDelayLine(
        taps=[
          TapProfile(
            delay_sec=0.0,
            power_db=0.0,
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
          TapProfile(
            delay_sec=0.001,  # 1 ms
            power_db=0.0,  # Equal power
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
        ]
      ),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def poor(
    snr_db: float = 5.0,
    doppler_hz: float = 2.0,
    freq_offset_hz: float = 10.0,
  ) -> list[ChannelImpairment]:
    """Poor HF conditions: heavy fading, low SNR.

    Typical for:
    - Very long range (> 4000 km)
    - Disturbed ionospheric conditions
    - High geomagnetic activity (K-index > 5)
    - Polar paths (aurora)

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 5 dB).
      doppler_hz: Maximum Doppler shift in Hz (default: 2 Hz).
      freq_offset_hz: Carrier frequency offset in Hz (default: 10 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      SSBFilter(low_cutoff_hz=300.0, high_cutoff_hz=2700.0),
      # ITU-R F.1487 "Poor" condition: 2 taps, 2ms delay spread
      TappedDelayLine(
        taps=[
          TapProfile(
            delay_sec=0.0,
            power_db=0.0,
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
          TapProfile(
            delay_sec=0.002,  # 2 ms
            power_db=0.0,  # Equal power
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
        ]
      ),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def nvis(
    snr_db: float = 15.0,
    doppler_hz: float = 0.2,
    freq_offset_hz: float = 1.0,
  ) -> list[ChannelImpairment]:
    """Near Vertical Incidence Skywave (NVIS) conditions.

    NVIS uses high-angle radiation (> 75°) for regional coverage (0-400 km).
    Characterized by:
    - Very low Doppler spread
    - Stable propagation
    - Typical frequencies: 2-10 MHz

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 15 dB).
      doppler_hz: Maximum Doppler shift in Hz (default: 0.2 Hz).
      freq_offset_hz: Carrier frequency offset in Hz (default: 1 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      SSBFilter(low_cutoff_hz=300.0, high_cutoff_hz=2700.0),
      RayleighFading(doppler_hz=doppler_hz),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]


# Pre-configured HF presets (ITU-R based)
hf.ITU_R_EXCELLENT = ChannelPreset(
  name="ITU-R Excellent",
  description="Excellent HF conditions: minimal fading, SNR=30dB",
  impairments=hf.excellent(),
  typical_sample_rate=hf.TYPICAL_SAMPLE_RATE,
)

hf.ITU_R_GOOD = ChannelPreset(
  name="ITU-R Good",
  description="Good HF conditions: light fading (0.5Hz), SNR=20dB",
  impairments=hf.good(),
  typical_sample_rate=hf.TYPICAL_SAMPLE_RATE,
)

hf.ITU_R_MODERATE = ChannelPreset(
  name="ITU-R Moderate",
  description="Moderate HF conditions: moderate fading (1Hz), SNR=12dB",
  impairments=hf.moderate(),
  typical_sample_rate=hf.TYPICAL_SAMPLE_RATE,
)

hf.ITU_R_POOR = ChannelPreset(
  name="ITU-R Poor",
  description="Poor HF conditions: heavy fading (2Hz), SNR=5dB",
  impairments=hf.poor(),
  typical_sample_rate=hf.TYPICAL_SAMPLE_RATE,
)

hf.NVIS = ChannelPreset(
  name="NVIS",
  description="Near Vertical Incidence Skywave: stable, low Doppler",
  impairments=hf.nvis(),
  typical_sample_rate=hf.TYPICAL_SAMPLE_RATE,
)


# =============================================================================
# VHF/UHF Channels - Line-of-Sight and Mobile
# =============================================================================


class vhf:  # noqa: N801
  """VHF/UHF (30-3000 MHz) channel presets for terrestrial mobile/fixed.

  VHF/UHF channels are characterized by:
  - Line-of-sight or ground reflection propagation
  - Mobile Doppler (10-200 Hz at vehicular speeds)
  - Rayleigh (NLOS) or Rician (LOS) fading
  - Typical sample rate: 48 kHz

  References:
    - 3GPP TS 36.101 (LTE channel models)
    - ITU-R M.1225 (Mobile channel models)
  """

  TYPICAL_SAMPLE_RATE: int = 48000  # Hz

  # Pre-configured presets (assigned below)
  FIXED_LOS: ChannelPreset
  MOBILE_URBAN: ChannelPreset
  MOBILE_RURAL: ChannelPreset
  MOBILE_PEDESTRIAN: ChannelPreset

  @staticmethod
  def fixed_los(
    snr_db: float = 25.0, freq_offset_hz: float = 5.0
  ) -> list[ChannelImpairment]:
    """Fixed line-of-sight: minimal fading, AWGN-dominated.

    Typical for:
    - Point-to-point links
    - Base station to fixed terminal
    - Clear line-of-sight paths

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 25 dB).
      freq_offset_hz: Carrier frequency offset in Hz (default: 5 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def mobile_urban(
    snr_db: float = 15.0,
    doppler_hz: float = 100.0,
    freq_offset_hz: float = 50.0,
  ) -> list[ChannelImpairment]:
    """Mobile urban: heavy Rayleigh fading, high Doppler.

    Typical for:
    - Urban vehicular (60 km/h at 900 MHz → ~50 Hz Doppler)
    - Dense multipath (buildings, reflections)
    - Non-line-of-sight (NLOS)

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 15 dB).
      doppler_hz: Maximum Doppler shift in Hz (default: 100 Hz).
      freq_offset_hz: Carrier frequency offset in Hz (default: 50 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      # 3GPP EVA (Extended Vehicular A) model - simplified to 5 main taps
      TappedDelayLine(
        taps=[
          TapProfile(
            delay_sec=0.0,
            power_db=0.0,
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
          TapProfile(
            delay_sec=30e-9,
            power_db=-1.5,
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
          TapProfile(
            delay_sec=150e-9,
            power_db=-1.4,
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
          TapProfile(
            delay_sec=310e-9,
            power_db=-3.6,
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
          TapProfile(
            delay_sec=710e-9,
            power_db=-9.1,
            fading_model=RayleighFading(doppler_hz=doppler_hz),
          ),
        ]
      ),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def mobile_rural(
    snr_db: float = 20.0,
    doppler_hz: float = 20.0,
    k_factor_db: float = 6.0,
    freq_offset_hz: float = 20.0,
  ) -> list[ChannelImpairment]:
    """Mobile rural: Rician fading with LOS component.

    Typical for:
    - Rural/suburban vehicular
    - Partial line-of-sight
    - Lower Doppler (lower speeds or frequencies)

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 20 dB).
      doppler_hz: Maximum Doppler shift in Hz (default: 20 Hz).
      k_factor_db: Rician K-factor in dB (default: 6 dB).
      freq_offset_hz: Carrier frequency offset in Hz (default: 20 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      RicianFading(doppler_hz=doppler_hz, k_factor_db=k_factor_db),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def mobile_pedestrian(
    snr_db: float = 18.0,
    doppler_hz: float = 5.0,
    freq_offset_hz: float = 10.0,
  ) -> list[ChannelImpairment]:
    """Mobile pedestrian: low Doppler, moderate fading.

    Typical for:
    - Pedestrian speeds (5 km/h at 2 GHz → ~9 Hz Doppler)
    - Indoor/outdoor transitions
    - Handheld devices

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 18 dB).
      doppler_hz: Maximum Doppler shift in Hz (default: 5 Hz).
      freq_offset_hz: Carrier frequency offset in Hz (default: 10 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      RayleighFading(doppler_hz=doppler_hz),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]


# Pre-configured VHF/UHF presets
vhf.FIXED_LOS = ChannelPreset(
  name="VHF Fixed LOS",
  description="Fixed line-of-sight: AWGN-dominated, SNR=25dB",
  impairments=vhf.fixed_los(),
  typical_sample_rate=vhf.TYPICAL_SAMPLE_RATE,
)

vhf.MOBILE_URBAN = ChannelPreset(
  name="VHF Mobile Urban",
  description="Urban mobile: Rayleigh fading (100Hz Doppler), SNR=15dB",
  impairments=vhf.mobile_urban(),
  typical_sample_rate=vhf.TYPICAL_SAMPLE_RATE,
)

vhf.MOBILE_RURAL = ChannelPreset(
  name="VHF Mobile Rural",
  description="Rural mobile: Rician fading (K=6dB, 20Hz Doppler), SNR=20dB",
  impairments=vhf.mobile_rural(),
  typical_sample_rate=vhf.TYPICAL_SAMPLE_RATE,
)

vhf.MOBILE_PEDESTRIAN = ChannelPreset(
  name="VHF Mobile Pedestrian",
  description="Pedestrian mobile: low Doppler (5Hz), SNR=18dB",
  impairments=vhf.mobile_pedestrian(),
  typical_sample_rate=vhf.TYPICAL_SAMPLE_RATE,
)


# =============================================================================
# Satellite Channels
# =============================================================================


class satellite:  # noqa: N801
  """Satellite channel presets for LEO, MEO, and GEO orbits.

  Satellite channels are characterized by:
  - High Doppler shift (LEO: up to ±3 kHz at UHF)
  - Possible Rician fading (strong LOS + multipath)
  - Long propagation delays (not modeled here)
  - Typical sample rate: 48 kHz or higher

  References:
    - ITU-R S.1428 (Satellite channel models)
    - CCSDS 401.0-B (Radio Frequency and Modulation Systems)
  """

  TYPICAL_SAMPLE_RATE: int = 48000  # Hz

  # Pre-configured presets (assigned below)
  LEO_CLEAR: ChannelPreset
  LEO_FADED: ChannelPreset
  GEO: ChannelPreset

  @staticmethod
  def leo_clear(
    snr_db: float = 20.0,
    doppler_hz: float = 10.0,
    k_factor_db: float = 10.0,
    freq_offset_hz: float = 100.0,
  ) -> list[ChannelImpairment]:
    """LEO satellite, clear conditions: strong LOS, high Doppler.

    Typical for:
    - Low Earth Orbit (400-2000 km altitude)
    - Clear sky, high elevation angle
    - Strong line-of-sight component

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 20 dB).
      doppler_hz: Doppler spread in Hz (default: 10 Hz).
      k_factor_db: Rician K-factor in dB (default: 10 dB).
      freq_offset_hz: Carrier frequency offset in Hz (default: 100 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      RicianFading(doppler_hz=doppler_hz, k_factor_db=k_factor_db),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def leo_faded(
    snr_db: float = 12.0,
    doppler_hz: float = 15.0,
    k_factor_db: float = 3.0,
    freq_offset_hz: float = 150.0,
  ) -> list[ChannelImpairment]:
    """LEO satellite, faded conditions: multipath, lower K-factor.

    Typical for:
    - Low elevation angles (< 20°)
    - Urban/suburban ground stations (multipath)
    - Partial obstruction

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 12 dB).
      doppler_hz: Doppler spread in Hz (default: 15 Hz).
      k_factor_db: Rician K-factor in dB (default: 3 dB).
      freq_offset_hz: Carrier frequency offset in Hz (default: 150 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      RicianFading(doppler_hz=doppler_hz, k_factor_db=k_factor_db),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def geo(
    snr_db: float = 15.0,
    freq_offset_hz: float = 10.0,
  ) -> list[ChannelImpairment]:
    """GEO satellite: minimal Doppler, AWGN-dominated.

    Typical for:
    - Geostationary orbit (35,786 km altitude)
    - Fixed ground station
    - Stable propagation path

    Args:
      snr_db: Signal-to-noise ratio in dB (default: 15 dB).
      freq_offset_hz: Carrier frequency offset in Hz (default: 10 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]


# Pre-configured satellite presets
satellite.LEO_CLEAR = ChannelPreset(
  name="LEO Clear",
  description="LEO satellite clear sky: Rician (K=10dB), SNR=20dB",
  impairments=satellite.leo_clear(),
  typical_sample_rate=satellite.TYPICAL_SAMPLE_RATE,
)

satellite.LEO_FADED = ChannelPreset(
  name="LEO Faded",
  description="LEO satellite faded: Rician (K=3dB), SNR=12dB",
  impairments=satellite.leo_faded(),
  typical_sample_rate=satellite.TYPICAL_SAMPLE_RATE,
)

satellite.GEO = ChannelPreset(
  name="GEO",
  description="GEO satellite: AWGN-dominated, SNR=15dB",
  impairments=satellite.geo(),
  typical_sample_rate=satellite.TYPICAL_SAMPLE_RATE,
)


# =============================================================================
# Earth-Moon-Earth (EME) Channels
# =============================================================================


class eme:  # noqa: N801
  """Earth-Moon-Earth (EME) moonbounce channel presets.

  EME channels are characterized by:
  - Extremely long path (768,000 km round trip)
  - Very high path loss (~250-270 dB at VHF/UHF)
  - Libration fading (Doppler spread from lunar motion: 0.5-2 Hz)
  - Polarization rotation (Faraday effect in ionosphere)
  - Typical frequencies: 144 MHz, 432 MHz, 1296 MHz
  - Typical sample rate: 48 kHz

  References:
    - ARRL Handbook: "EME Communications"
    - W1GHZ: "Microwave Antenna Book"
    - VK3UM: "EME Calculator"

  Note: Path loss is not modeled here (handled by link budget).
  """

  TYPICAL_SAMPLE_RATE: int = 48000  # Hz

  # Pre-configured presets (assigned below)
  SMOOTH_MOON: ChannelPreset
  ROUGH_MOON: ChannelPreset
  DEGRADED: ChannelPreset

  @staticmethod
  def smooth_moon(
    snr_db: float = -3.0,
    doppler_hz: float = 0.5,
    freq_offset_hz: float = 200.0,
  ) -> list[ChannelImpairment]:
    """EME with smooth lunar surface reflection.

    Typical for:
    - Specular reflection from smooth mare regions
    - Low libration fading
    - Optimal moon position (low declination rate)

    Args:
      snr_db: Signal-to-noise ratio in dB (default: -3 dB, typical for EME).
      doppler_hz: Libration Doppler spread in Hz (default: 0.5 Hz).
      freq_offset_hz: Carrier frequency offset in Hz (default: 200 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      RicianFading(doppler_hz=doppler_hz, k_factor_db=6.0),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def rough_moon(
    snr_db: float = -5.0,
    doppler_hz: float = 1.5,
    freq_offset_hz: float = 200.0,
  ) -> list[ChannelImpairment]:
    """EME with rough lunar surface scattering.

    Typical for:
    - Diffuse scattering from rough highland regions
    - Higher libration fading
    - Rapid moon declination changes

    Args:
      snr_db: Signal-to-noise ratio in dB (default: -5 dB).
      doppler_hz: Libration Doppler spread in Hz (default: 1.5 Hz).
      freq_offset_hz: Carrier frequency offset in Hz (default: 200 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      RayleighFading(doppler_hz=doppler_hz),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]

  @staticmethod
  def degraded(
    snr_db: float = -8.0,
    doppler_hz: float = 2.0,
    freq_offset_hz: float = 250.0,
  ) -> list[ChannelImpairment]:
    """EME with degraded conditions.

    Typical for:
    - Low moon elevation (< 10°)
    - High atmospheric noise
    - Suboptimal antenna pointing
    - Ionospheric disturbances (Faraday rotation)

    Args:
      snr_db: Signal-to-noise ratio in dB (default: -8 dB).
      doppler_hz: Libration Doppler spread in Hz (default: 2 Hz).
      freq_offset_hz: Carrier frequency offset in Hz (default: 250 Hz).

    Returns:
      List of channel impairments.
    """
    return [
      RayleighFading(doppler_hz=doppler_hz),
      FreqOffset(offset_hz=freq_offset_hz),
      AWGN(snr_db=snr_db),
    ]


# Pre-configured EME presets
eme.SMOOTH_MOON = ChannelPreset(
  name="EME Smooth Moon",
  description="EME smooth surface: Rician (K=6dB), libration 0.5Hz, SNR=-3dB",
  impairments=eme.smooth_moon(),
  typical_sample_rate=eme.TYPICAL_SAMPLE_RATE,
)

eme.ROUGH_MOON = ChannelPreset(
  name="EME Rough Moon",
  description="EME rough surface: Rayleigh, libration 1.5Hz, SNR=-5dB",
  impairments=eme.rough_moon(),
  typical_sample_rate=eme.TYPICAL_SAMPLE_RATE,
)

eme.DEGRADED = ChannelPreset(
  name="EME Degraded",
  description="EME degraded: heavy fading, libration 2Hz, SNR=-8dB",
  impairments=eme.degraded(),
  typical_sample_rate=eme.TYPICAL_SAMPLE_RATE,
)


# =============================================================================
# AWGN-Only Channels (Baseline/Testing)
# =============================================================================


class awgn:  # noqa: N801
  """AWGN-only channel presets for baseline testing.

  Pure additive white Gaussian noise without fading or other impairments.
  Useful for:
  - Baseline performance measurement
  - Codec testing
  - SNR threshold determination
  """

  TYPICAL_SAMPLE_RATE: int = 48000  # Hz (arbitrary, works at any rate)

  # Pre-configured presets (assigned below)
  SNR_30DB: ChannelPreset
  SNR_20DB: ChannelPreset
  SNR_10DB: ChannelPreset
  SNR_0DB: ChannelPreset

  @staticmethod
  def only(snr_db: float) -> list[ChannelImpairment]:
    """Pure AWGN channel at specified SNR.

    Args:
      snr_db: Signal-to-noise ratio in dB.

    Returns:
      List containing only AWGN impairment.
    """
    return [AWGN(snr_db=snr_db)]


# Pre-configured AWGN presets at common SNR levels
awgn.SNR_30DB = ChannelPreset(
  name="AWGN 30dB",
  description="Pure AWGN: SNR=30dB (excellent)",
  impairments=awgn.only(30.0),
  typical_sample_rate=awgn.TYPICAL_SAMPLE_RATE,
)

awgn.SNR_20DB = ChannelPreset(
  name="AWGN 20dB",
  description="Pure AWGN: SNR=20dB (good)",
  impairments=awgn.only(20.0),
  typical_sample_rate=awgn.TYPICAL_SAMPLE_RATE,
)

awgn.SNR_10DB = ChannelPreset(
  name="AWGN 10dB",
  description="Pure AWGN: SNR=10dB (moderate)",
  impairments=awgn.only(10.0),
  typical_sample_rate=awgn.TYPICAL_SAMPLE_RATE,
)

awgn.SNR_0DB = ChannelPreset(
  name="AWGN 0dB",
  description="Pure AWGN: SNR=0dB (poor)",
  impairments=awgn.only(0.0),
  typical_sample_rate=awgn.TYPICAL_SAMPLE_RATE,
)
