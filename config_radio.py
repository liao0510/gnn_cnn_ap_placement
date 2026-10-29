"""Noise-limited HE access and AP peering; constant power spectral density.

Inter-AP interference and inter-AP traffic scheduling are outside this model.
OFDMA uses equal-size, nonoverlapping RUs and round robin when oversubscribed.
"""
from dataclasses import dataclass
import math

ACCESS_FREQUENCY_GHZ = 5.210
ACCESS_BANDWIDTH_MHZ = 80
TX_POWER_DBM = 20.0
NOISE_FIGURE_DB = 7.0
MESH_SNR_THRESHOLD_DB = 4.0
CLIENT_SNR_THRESHOLD_DB = 4.0
COVERAGE_THRESHOLD_DBM = -60.0
GRID_SIZE_METERS = 0.1
MAC_EFFICIENCY = 0.7
SPATIAL_STREAMS = 1
GUARD_INTERVAL_US = 0.8
OFDMA_SCHEDULER = 'equal_ru_round_robin'


@dataclass(frozen=True)
class ChannelConfig:
    bandwidth_mhz: int = ACCESS_BANDWIDTH_MHZ
    noise_figure_db: float = NOISE_FIGURE_DB
    mesh_snr_threshold_db: float = MESH_SNR_THRESHOLD_DB
    client_snr_threshold_db: float = CLIENT_SNR_THRESHOLD_DB
    spatial_streams: int = SPATIAL_STREAMS
    guard_interval_us: float = GUARD_INTERVAL_US

    def __post_init__(self):
        if self.bandwidth_mhz not in (20, 40, 80, 160):
            raise ValueError('HE bandwidth must be 20, 40, 80 or 160 MHz')
        if self.guard_interval_us not in (0.8, 1.6, 3.2):
            raise ValueError('HE guard interval must be 0.8, 1.6 or 3.2 us')
        if not isinstance(self.spatial_streams, int) or not 1 <= self.spatial_streams <= 8:
            raise ValueError('spatial_streams must be an integer from 1 to 8')
        if not all(math.isfinite(v) for v in (self.noise_figure_db,
                   self.mesh_snr_threshold_db, self.client_snr_threshold_db)):
            raise ValueError('Noise figure and SNR thresholds must be finite')


def snapshot():
    return {key: value for key, value in globals().items() if key.isupper()}


def channel_config(settings=None):
    values = snapshot() if settings is None else settings
    if values['OFDMA_SCHEDULER'] != 'equal_ru_round_robin':
        raise ValueError('Unsupported OFDMA scheduler')
    return ChannelConfig(bandwidth_mhz=values['ACCESS_BANDWIDTH_MHZ'],
                         noise_figure_db=values['NOISE_FIGURE_DB'],
                         mesh_snr_threshold_db=values['MESH_SNR_THRESHOLD_DB'],
                         client_snr_threshold_db=values['CLIENT_SNR_THRESHOLD_DB'],
                         spatial_streams=values['SPATIAL_STREAMS'],
                         guard_interval_us=values['GUARD_INTERVAL_US'])
