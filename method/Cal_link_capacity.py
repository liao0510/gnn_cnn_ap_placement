"""SNR -> assumed MCS threshold -> HE rate at the allocated RU size.

Thresholds approximate receiver performance, rather than normative PER curves.
Rates use HE data subcarriers and a 12.8 us symbol plus the guard interval.
"""
import math
import config_radio as radio

# MCS, constellation bits, coding rate, assumed minimum SNR (dB).
MCS_TABLE = (
    (0, 1, 1/2, 4), (1, 2, 1/2, 7), (2, 2, 3/4, 10),
    (3, 4, 1/2, 13), (4, 4, 3/4, 16), (5, 6, 2/3, 20),
    (6, 6, 3/4, 23), (7, 6, 5/6, 25), (8, 8, 3/4, 28),
    (9, 8, 5/6, 31), (10, 10, 3/4, 34), (11, 10, 5/6, 37),
)
RU_DATA_SUBCARRIERS = {26: 24, 52: 48, 106: 102, 242: 234,
                       484: 468, 996: 980, 1992: 1960}
FULL_CHANNEL_RU = {20: 242, 40: 484, 80: 996, 160: 1992}
# Homogeneous legal HE layouts; mixed sizes and MU-MIMO are omitted.
RU_LAYOUTS = {
    20: {26: 9, 52: 4, 106: 2, 242: 1},
    40: {26: 18, 52: 8, 106: 4, 242: 2, 484: 1},
    80: {26: 37, 52: 16, 106: 8, 242: 4, 484: 2, 996: 1},
    160: {26: 74, 52: 32, 106: 16, 242: 8, 484: 4, 996: 2, 1992: 1},
}


def calculate_link_capacity(snr_db, mac_efficiency=None, *, ru_tones=None,
                            bandwidth_mhz=None, spatial_streams=None,
                            guard_interval_us=None):
    efficiency = radio.MAC_EFFICIENCY if mac_efficiency is None else mac_efficiency
    bandwidth = radio.ACCESS_BANDWIDTH_MHZ if bandwidth_mhz is None else bandwidth_mhz
    streams = radio.SPATIAL_STREAMS if spatial_streams is None else spatial_streams
    gi = radio.GUARD_INTERVAL_US if guard_interval_us is None else guard_interval_us
    if not math.isfinite(efficiency) or not 0 <= efficiency <= 1:
        raise ValueError('MAC efficiency must be finite and between zero and one')
    if bandwidth not in FULL_CHANNEL_RU:
        raise ValueError('Unsupported HE channel bandwidth')
    if not isinstance(streams, int) or not 1 <= streams <= 8 or gi not in (0.8, 1.6, 3.2):
        raise ValueError('Invalid HE spatial streams or guard interval')
    tones = FULL_CHANNEL_RU[bandwidth] if ru_tones is None else ru_tones
    if tones not in RU_LAYOUTS[bandwidth]:
        raise ValueError('RU does not fit the selected channel bandwidth')
    selected = None
    if snr_db is not None and snr_db is not False and math.isfinite(float(snr_db)):
        selected = next((row for row in reversed(MCS_TABLE) if float(snr_db) >= row[3]), None)
    phy_rate = 0.0 if selected is None else (
        RU_DATA_SUBCARRIERS[tones] * selected[1] * selected[2] * streams / (12.8 + gi))
    return {'mcs': None if selected is None else selected[0], 'ru_tones': tones,
            'data_subcarriers': RU_DATA_SUBCARRIERS[tones],
            'phy_rate_mbps': phy_rate, 'link_capacity_mbps': phy_rate * efficiency}
