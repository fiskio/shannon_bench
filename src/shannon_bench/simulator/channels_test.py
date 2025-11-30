"""Tests for standard channel library presets."""

import numpy as np
import pytest

from shannon_bench.simulator.channels import awgn, eme, hf, satellite, vhf
from shannon_bench.simulator.transmission_system import (
  AWGN,
  ChannelImpairment,
  ChannelSimulator,
  FreqOffset,
  RayleighFading,
  RicianFading,
  SSBFilter,
  TappedDelayLine,
)


class TestHFChannels:
  """Tests for HF channel presets."""

  @pytest.mark.parametrize(
    "factory_func",
    [
      hf.excellent,
      hf.good,
      hf.moderate,
      hf.poor,
      hf.nvis,
    ],
  )
  def test_factory_returns_valid_impairments(self, factory_func) -> None:
    """Test that factory functions return valid impairment lists."""
    impairments = factory_func()
    assert isinstance(impairments, list)
    assert len(impairments) > 0
    assert all(isinstance(imp, ChannelImpairment) for imp in impairments)

  @pytest.mark.parametrize(
    "factory_func",
    [
      hf.excellent,
      hf.good,
      hf.moderate,
      hf.poor,
      hf.nvis,
    ],
  )
  def test_impairments_are_sorted_by_stage(self, factory_func) -> None:
    """Test that impairments are in canonical order."""
    impairments = factory_func()
    stages = [imp.stage for imp in impairments]
    assert stages == sorted(stages), f"Stages not sorted: {stages}"

  def test_excellent_has_no_fading(self) -> None:
    """Test that excellent conditions have no fading."""
    impairments = hf.excellent()
    fading_types = (RayleighFading, RicianFading)
    assert not any(isinstance(imp, fading_types) for imp in impairments)

  def test_poor_has_fading(self) -> None:
    """Test that poor conditions include fading."""
    impairments = hf.poor()
    fading_types = (RayleighFading, RicianFading, TappedDelayLine)
    assert any(isinstance(imp, fading_types) for imp in impairments)

  def test_all_have_ssb_filter(self) -> None:
    """Test that all HF presets include SSB filter."""
    for factory_func in [hf.excellent, hf.good, hf.moderate, hf.poor, hf.nvis]:
      impairments = factory_func()
      assert any(isinstance(imp, SSBFilter) for imp in impairments)

  def test_all_have_awgn(self) -> None:
    """Test that all HF presets include AWGN."""
    for factory_func in [hf.excellent, hf.good, hf.moderate, hf.poor, hf.nvis]:
      impairments = factory_func()
      assert any(isinstance(imp, AWGN) for imp in impairments)

  @pytest.mark.parametrize(
    ("factory_func", "expected_min_snr"),
    [
      (hf.excellent, 25.0),
      (hf.good, 15.0),
      (hf.moderate, 10.0),
      (hf.poor, 3.0),
    ],
  )
  def test_snr_decreases_with_quality(self, factory_func, expected_min_snr) -> None:
    """Test that SNR decreases as conditions worsen."""
    impairments = factory_func()
    awgn_imp = next(imp for imp in impairments if isinstance(imp, AWGN))
    assert awgn_imp.snr_db >= expected_min_snr

  def test_custom_parameters(self) -> None:
    """Test that custom parameters are respected."""
    impairments = hf.poor(snr_db=10.0, doppler_hz=3.0, freq_offset_hz=20.0)

    awgn_imp = next(imp for imp in impairments if isinstance(imp, AWGN))
    assert awgn_imp.snr_db == 10.0

    tdl_imp = next(imp for imp in impairments if isinstance(imp, TappedDelayLine))
    # Check first tap's fading model
    assert tdl_imp.taps[0].fading_model.doppler_hz == 3.0

    offset_imp = next(imp for imp in impairments if isinstance(imp, FreqOffset))
    assert offset_imp.offset_hz == 20.0

  def test_preset_constants_exist(self) -> None:
    """Test that pre-configured preset constants exist."""
    assert hasattr(hf, "ITU_R_EXCELLENT")
    assert hasattr(hf, "ITU_R_GOOD")
    assert hasattr(hf, "ITU_R_MODERATE")
    assert hasattr(hf, "ITU_R_POOR")
    assert hasattr(hf, "NVIS")

  def test_preset_constants_have_metadata(self) -> None:
    """Test that preset constants have proper metadata."""
    preset = hf.ITU_R_GOOD
    assert preset.name
    assert preset.description
    assert preset.typical_sample_rate == hf.TYPICAL_SAMPLE_RATE
    assert len(preset.impairments) > 0

  def test_hf_taps_are_resolved(self) -> None:
    """Test that HF taps result in distinct delays at typical sample rate."""
    # Check "Good" condition (0.5 ms delay)
    impairments = hf.good()
    tdl = next(imp for imp in impairments if isinstance(imp, TappedDelayLine))
    
    # Calculate expected delay in samples at 8 kHz
    # 0.5 ms * 8000 Hz = 4 samples
    delay_sec = tdl.taps[1].delay_sec
    delay_samples = int(round(delay_sec * hf.TYPICAL_SAMPLE_RATE))
    
    assert delay_samples > 0, "HF taps collapsed to flat fading!"
    assert delay_samples == 4


class TestVHFChannels:
  """Tests for VHF/UHF channel presets."""

  @pytest.mark.parametrize(
    "factory_func",
    [
      vhf.fixed_los,
      vhf.mobile_urban,
      vhf.mobile_rural,
      vhf.mobile_pedestrian,
    ],
  )
  def test_factory_returns_valid_impairments(self, factory_func) -> None:
    """Test that factory functions return valid impairment lists."""
    impairments = factory_func()
    assert isinstance(impairments, list)
    assert len(impairments) > 0
    assert all(isinstance(imp, ChannelImpairment) for imp in impairments)

  def test_fixed_los_has_no_fading(self) -> None:
    """Test that fixed LOS has no fading."""
    impairments = vhf.fixed_los()
    fading_types = (RayleighFading, RicianFading)
    assert not any(isinstance(imp, fading_types) for imp in impairments)

  def test_mobile_urban_has_rayleigh(self) -> None:
    """Test that mobile urban uses Rayleigh fading (NLOS)."""
    impairments = vhf.mobile_urban()
    # Should use TappedDelayLine with Rayleigh fading on taps
    tdl = next(imp for imp in impairments if isinstance(imp, TappedDelayLine))
    assert any(
      isinstance(tap.fading_model, RayleighFading) for tap in tdl.taps
    )

  def test_mobile_rural_has_rician(self) -> None:
    """Test that mobile rural uses Rician fading (LOS component)."""
    impairments = vhf.mobile_rural()
    assert any(isinstance(imp, RicianFading) for imp in impairments)

  @pytest.mark.parametrize(
    ("factory_func", "expected_max_doppler"),
    [
      (vhf.mobile_pedestrian, 10.0),
      (vhf.mobile_rural, 50.0),
      (vhf.mobile_urban, 150.0),
    ],
  )
  def test_doppler_increases_with_mobility(
    self, factory_func, expected_max_doppler
  ) -> None:
    """Test that Doppler increases with mobility level."""
    impairments = factory_func()
    fading_types = (RayleighFading, RicianFading)
    fading_imps = [imp for imp in impairments if isinstance(imp, fading_types)]
    tdl_imps = [imp for imp in impairments if isinstance(imp, TappedDelayLine)]

    if fading_imps:
      assert fading_imps[0].doppler_hz <= expected_max_doppler
    elif tdl_imps:
      # Check first tap of TDL
      first_tap_fading = tdl_imps[0].taps[0].fading_model
      if first_tap_fading:
        assert first_tap_fading.doppler_hz <= expected_max_doppler

  def test_preset_constants_exist(self) -> None:
    """Test that pre-configured preset constants exist."""
    assert hasattr(vhf, "FIXED_LOS")
    assert hasattr(vhf, "MOBILE_URBAN")
    assert hasattr(vhf, "MOBILE_RURAL")
    assert hasattr(vhf, "MOBILE_PEDESTRIAN")


class TestSatelliteChannels:
  """Tests for satellite channel presets."""

  @pytest.mark.parametrize(
    "factory_func",
    [
      satellite.leo_clear,
      satellite.leo_faded,
      satellite.geo,
    ],
  )
  def test_factory_returns_valid_impairments(self, factory_func) -> None:
    """Test that factory functions return valid impairment lists."""
    impairments = factory_func()
    assert isinstance(impairments, list)
    assert len(impairments) > 0
    assert all(isinstance(imp, ChannelImpairment) for imp in impairments)

  def test_leo_clear_has_high_k_factor(self) -> None:
    """Test that LEO clear has strong LOS (high K-factor)."""
    impairments = satellite.leo_clear()
    rician_imp = next(imp for imp in impairments if isinstance(imp, RicianFading))
    assert rician_imp.k_factor_db >= 8.0

  def test_leo_faded_has_lower_k_factor(self) -> None:
    """Test that LEO faded has weaker LOS (lower K-factor)."""
    impairments = satellite.leo_faded()
    rician_imp = next(imp for imp in impairments if isinstance(imp, RicianFading))
    assert rician_imp.k_factor_db < 8.0

  def test_geo_has_no_fading(self) -> None:
    """Test that GEO has no fading (stable path)."""
    impairments = satellite.geo()
    fading_types = (RayleighFading, RicianFading)
    assert not any(isinstance(imp, fading_types) for imp in impairments)

  def test_preset_constants_exist(self) -> None:
    """Test that pre-configured preset constants exist."""
    assert hasattr(satellite, "LEO_CLEAR")
    assert hasattr(satellite, "LEO_FADED")
    assert hasattr(satellite, "GEO")


class TestEMEChannels:
  """Tests for Earth-Moon-Earth channel presets."""

  @pytest.mark.parametrize(
    "factory_func",
    [
      eme.smooth_moon,
      eme.rough_moon,
      eme.degraded,
    ],
  )
  def test_factory_returns_valid_impairments(self, factory_func) -> None:
    """Test that factory functions return valid impairment lists."""
    impairments = factory_func()
    assert isinstance(impairments, list)
    assert len(impairments) > 0
    assert all(isinstance(imp, ChannelImpairment) for imp in impairments)

  def test_smooth_moon_has_rician(self) -> None:
    """Test that smooth moon uses Rician fading (specular reflection)."""
    impairments = eme.smooth_moon()
    assert any(isinstance(imp, RicianFading) for imp in impairments)

  def test_rough_moon_has_rayleigh(self) -> None:
    """Test that rough moon uses Rayleigh fading (diffuse scattering)."""
    impairments = eme.rough_moon()
    assert any(isinstance(imp, RayleighFading) for imp in impairments)

  @pytest.mark.parametrize(
    "factory_func",
    [
      eme.smooth_moon,
      eme.rough_moon,
      eme.degraded,
    ],
  )
  def test_low_doppler_libration(self, factory_func) -> None:
    """Test that EME has low Doppler (libration fading)."""
    impairments = factory_func()
    fading_types = (RayleighFading, RicianFading)
    fading_imps = [imp for imp in impairments if isinstance(imp, fading_types)]

    if fading_imps:
      # Libration fading is typically < 3 Hz
      assert fading_imps[0].doppler_hz <= 3.0

  @pytest.mark.parametrize(
    "factory_func",
    [
      eme.smooth_moon,
      eme.rough_moon,
      eme.degraded,
    ],
  )
  def test_low_snr(self, factory_func) -> None:
    """Test that EME has very low SNR (high path loss)."""
    impairments = factory_func()
    awgn_imp = next(imp for imp in impairments if isinstance(imp, AWGN))
    # EME typically has negative SNR
    assert awgn_imp.snr_db <= 0.0

  def test_preset_constants_exist(self) -> None:
    """Test that pre-configured preset constants exist."""
    assert hasattr(eme, "SMOOTH_MOON")
    assert hasattr(eme, "ROUGH_MOON")
    assert hasattr(eme, "DEGRADED")


class TestAWGNChannels:
  """Tests for AWGN-only channel presets."""

  def test_awgn_only_returns_single_impairment(self) -> None:
    """Test that AWGN-only returns just AWGN."""
    impairments = awgn.only(15.0)
    assert len(impairments) == 1
    assert isinstance(impairments[0], AWGN)
    assert impairments[0].snr_db == 15.0

  def test_preset_constants_exist(self) -> None:
    """Test that pre-configured preset constants exist."""
    assert hasattr(awgn, "SNR_30DB")
    assert hasattr(awgn, "SNR_20DB")
    assert hasattr(awgn, "SNR_10DB")
    assert hasattr(awgn, "SNR_0DB")

  @pytest.mark.parametrize(
    ("preset", "expected_snr"),
    [
      (awgn.SNR_30DB, 30.0),
      (awgn.SNR_20DB, 20.0),
      (awgn.SNR_10DB, 10.0),
      (awgn.SNR_0DB, 0.0),
    ],
  )
  def test_preset_snr_values(self, preset, expected_snr) -> None:
    """Test that preset SNR values are correct."""
    awgn_imp = preset.impairments[0]
    assert isinstance(awgn_imp, AWGN)
    assert awgn_imp.snr_db == expected_snr


class TestChannelIntegration:
  """Integration tests: apply presets to actual signals."""

  @pytest.fixture
  def test_signal(self) -> np.ndarray:
    """Create a test signal (1 kHz tone, 100ms)."""
    sample_rate = 8000
    duration = 0.1
    t = np.arange(int(sample_rate * duration)) / sample_rate
    # Complex baseband: 1 kHz tone
    return np.exp(2j * np.pi * 1000 * t).astype(np.complex64)

  @pytest.mark.parametrize(
    ("preset_class", "factory_func"),
    [
      (hf, hf.excellent),
      (hf, hf.good),
      (hf, hf.moderate),
      (hf, hf.poor),
      (hf, hf.nvis),
      (vhf, vhf.fixed_los),
      (vhf, vhf.mobile_urban),
      (vhf, vhf.mobile_rural),
      (vhf, vhf.mobile_pedestrian),
      (satellite, satellite.leo_clear),
      (satellite, satellite.leo_faded),
      (satellite, satellite.geo),
      (eme, eme.smooth_moon),
      (eme, eme.rough_moon),
      (eme, eme.degraded),
      (awgn, awgn.only),
    ],
  )
  def test_preset_applies_to_signal(
    self, test_signal, preset_class, factory_func
  ) -> None:
    """Test that presets can be applied to a signal without errors."""
    # Get impairments
    impairments = factory_func(10.0) if factory_func == awgn.only else factory_func()

    # Create channel simulator
    channel = ChannelSimulator(
      impairments=impairments,
      sample_rate=preset_class.TYPICAL_SAMPLE_RATE,
    )

    # Apply to signal
    output = channel.apply(test_signal)

    # Verify output
    assert output.shape == test_signal.shape
    assert output.dtype == np.complex64
    assert not np.any(np.isnan(output))
    assert not np.any(np.isinf(output))

  @pytest.mark.parametrize(
    "preset",
    [
      hf.ITU_R_EXCELLENT,
      hf.ITU_R_GOOD,
      hf.ITU_R_MODERATE,
      hf.ITU_R_POOR,
      hf.NVIS,
      vhf.FIXED_LOS,
      vhf.MOBILE_URBAN,
      vhf.MOBILE_RURAL,
      vhf.MOBILE_PEDESTRIAN,
      satellite.LEO_CLEAR,
      satellite.LEO_FADED,
      satellite.GEO,
      eme.SMOOTH_MOON,
      eme.ROUGH_MOON,
      eme.DEGRADED,
      awgn.SNR_30DB,
      awgn.SNR_20DB,
      awgn.SNR_10DB,
      awgn.SNR_0DB,
    ],
  )
  def test_preset_constant_applies_to_signal(self, test_signal, preset) -> None:
    """Test that preset constants can be applied to a signal."""
    channel = ChannelSimulator(
      impairments=preset.impairments,
      sample_rate=preset.typical_sample_rate,
    )

    output = channel.apply(test_signal)

    assert output.shape == test_signal.shape
    assert output.dtype == np.complex64
    assert not np.any(np.isnan(output))
    assert not np.any(np.isinf(output))




class TestTypicalSampleRates:
  """Tests for typical sample rate recommendations."""

  def test_hf_sample_rate(self) -> None:
    """Test that HF typical sample rate is appropriate for SSB."""
    # SSB bandwidth is ~3 kHz, Nyquist requires > 6 kHz
    assert hf.TYPICAL_SAMPLE_RATE >= 6000
    assert hf.TYPICAL_SAMPLE_RATE == 8000  # Standard audio rate

  def test_vhf_sample_rate(self) -> None:
    """Test that VHF typical sample rate is higher than HF."""
    assert vhf.TYPICAL_SAMPLE_RATE > hf.TYPICAL_SAMPLE_RATE
    assert vhf.TYPICAL_SAMPLE_RATE == 48000  # Standard high-quality audio

  def test_satellite_sample_rate(self) -> None:
    """Test that satellite typical sample rate is appropriate."""
    assert satellite.TYPICAL_SAMPLE_RATE >= 48000

  def test_eme_sample_rate(self) -> None:
    """Test that EME typical sample rate is appropriate."""
    assert eme.TYPICAL_SAMPLE_RATE >= 48000
