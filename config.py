"""AP/client placement settings. Physical and OFDMA settings: config_radio.py."""
ENVIRONMENT_VERSION = 'ap_mesh_snr_ofdma_v1'
MAP_DIR = 'maps_200_200'
EPISODES = 4000
LR = 0.0001
GAMMA = 0.99
N_CLIENT = 10
AP_AMOUNT = 5                    # Total APs, all placed by the policy.
CLIENT_DEMAND = 100.0
MAX_NODES = 64
RANDOM_SEED = None
CRITIC_LOSS_WEIGHT = 0.5
GRADIENT_CLIP_NORM = 1.0
COVERAGE_REWARD_WEIGHT = 0.3
THROUGHPUT_REWARD_WEIGHT = 0.6
CONNECTIVITY_REWARD_WEIGHT = 0.1
COVERAGE_GAIN_SCALE = 0.25
THROUGHPUT_GAIN_SCALE = 0.25
MODEL_DIR = 'model'
RUN_PREFIX = 'AP_MESH_OFDMA'
MODEL_FILENAME = 'model.keras'
CHECKPOINT_INTERVAL = 100
SHOW_TRAINING_PLOTS = True
PRINT_MODEL_SUMMARY = True
RSSI_RAY_STEPS = 128
RSSI_ROW_CHUNK = 16
PREDICT_MODEL_PATH = None
PREDICT_MAP_PATH = 'maps_200_200/map_0000.json'
PREDICT_AP_COUNT = 7
PREDICT_CLIENT_COUNT = 10
PREDICT_CLIENT_DEMAND_MBPS = 100.0
PREDICT_CLIENT_POSITIONS = None
PREDICT_CLIENT_DEMANDS = None
PREDICT_RANDOM_SEED = 42
PREDICT_OUTPUT_DIR = 'prediction_results'
SHOW_PREDICTION_PLOTS = True
PRINT_PREDICTION_MODEL_SUMMARY = False


def snapshot():
    return {key: value for key, value in globals().items() if key.isupper()}


def reward_settings():
    return {key: value for key, value in snapshot().items()
            if key.endswith('_REWARD_WEIGHT') or key.endswith('_GAIN_SCALE')}
