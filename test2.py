"""Full-band SNR and HE rate sanity check."""
import config_radio as radio
from method.Cal_SNR import calculate_noise_dbm
from method.Cal_link_capacity import calculate_link_capacity

if __name__ == '__main__':
    snr = -60.0 - calculate_noise_dbm(radio.channel_config())
    print('SNR (dB):', snr)
    print(calculate_link_capacity(snr))
