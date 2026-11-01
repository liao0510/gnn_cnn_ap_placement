"""AP/client placement settings. Physical and OFDMA settings: config_radio.py."""
ENVIRONMENT_VERSION = 'ap_mesh_snr_ofdma_v1'
MODEL_ARCHITECTURE = 'gnn_cnn_multiscale_v2'
MAP_DIR = 'maps_200_200'
EPISODES = 2000
LR = 0.0001                     # PPO 的 Adam learning rate。
GAMMA = 0.99                    # 折扣率：越接近 1，越重視後續 AP 的 reward。
N_CLIENT = 10
AP_AMOUNT = 5                    # Total APs, all placed by the policy.
CLIENT_DEMAND = 100.0
MAX_NODES = 64
RANDOM_SEED = None
# PPO-Clip 訓練參數（不影響 reward、channel model、Mesh 或 OFDMA 定義）。
TRAINING_ALGORITHM = 'ppo_clip_v1'  # 記錄到實驗快照；本訓練程式使用 PPO-Clip。
ROLLOUT_EPISODES = 32           # 每批收集的完整 episodes；5 AP 時為 160 transitions。
PPO_MINIBATCH_SIZE = 40         # 每次 optimizer 更新的 transitions 數；最後不足一批仍更新。
PPO_EPOCHS = 5                  # 同一 rollout 重複訓練幾輪；每輪重新打亂 mini-batches。
PPO_CLIP_EPSILON = 0.2          # PPO 機率比 clipping 範圍 [1-epsilon, 1+epsilon]。
GAE_LAMBDA = 0.95               # GAE 優勢估計的 lambda；1 為完整 episode 的 MC return 減 value。
PPO_ENTROPY_COEF = 0.01         # Loss 減去此係數乘 entropy，鼓勵探索；0 關閉此項。
PPO_VALUE_CLIP_EPSILON = None   # None: 一般 value MSE；正數: 相對舊 value 的絕對 clipping 幅度。
PPO_TARGET_KL = 0.02            # None: 跑滿 epochs；正數: mini-batch approximate KL > 1.5 倍時提早停止。
PPO_SHUFFLE_MINIBATCHES = True  # 每個 PPO epoch 重新打亂 rollout，False 可用於重現/除錯。
PPO_ADAM_EPSILON = 1e-5         # Adam 分母的數值穩定常數，與 PPO clipping epsilon 不同。
CRITIC_LOSS_WEIGHT = 0.5        # 總 loss = actor + 此係數 * value MSE - entropy 項。
GRADIENT_CLIP_NORM = 1.0        # 每個 mini-batch 累積完成後才限制梯度 global norm。
TRAIN_MICROBATCH_SIZE = 2       # 一次前後向計算的 transitions 數；滿一個 PPO mini-batch 才更新。
NORMALIZE_ADVANTAGES = True     # 整批 rollout 的 actor advantage 正規化一次，所有 epochs 共用。
VALIDATION_INTERVAL = 100       # 到達此間隔後，在下一次更新完成時驗證；最後一定驗證。
VALIDATION_EPISODES = 20        # 固定的地圖/client 情境數。
VALIDATION_RANDOM_SEED = 20261030
VALIDATION_MAP_DIR = None       # None: 自動保留部分地圖；也可指定獨立驗證地圖資料夾。
VALIDATION_MAP_FRACTION = 0.2    # 自動保留地圖比例；只有一張圖時使用不同的固定 clients。
BEST_MODEL_FILENAME = 'best_model.keras'
COVERAGE_REWARD_WEIGHT = 0.2
THROUGHPUT_REWARD_WEIGHT = 0.7
CONNECTIVITY_REWARD_WEIGHT = 0.1
COVERAGE_GAIN_SCALE = 0.25
THROUGHPUT_GAIN_SCALE = 0.25
MODEL_DIR = 'model'
RUN_PREFIX = 'AP_MESH_OFDMA'
MODEL_FILENAME = 'model.keras'
CHECKPOINT_INTERVAL = 100
SHOW_TRAINING_PLOTS = False
PRINT_MODEL_SUMMARY = True
RSSI_RAY_STEPS = 128
RSSI_ROW_CHUNK = 16
PREDICT_MODEL_PATH = None        # None: 最新完成訓練的 best model；舊訓練無 best 時用 final。
PREDICT_MAP_PATH = 'maps_200_200/map_0000.json'
PREDICT_AP_COUNT = 5
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
