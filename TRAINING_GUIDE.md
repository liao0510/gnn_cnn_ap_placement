# PPO-Clip 訓練與最佳模型

目前 AP-AP 使用 SNR、AP-client 使用 SINR；新環境與重新訓練要求見 [SINR_MODEL.md](SINR_MODEL.md)。
以下 PPO 流程不變，舊 SNR 模型不能直接當成新環境模型載入。

目前模型架構為 `gnn_cnn_client_ap_attention_v4`，已新增直接的 AP 空間融合；詳見 `AP_ATTENTION.md`。v3 權重保留，但不能直接作為 v4 權重載入，須重新訓練。

在專案目錄執行 `python GNN_CNN_DRL_train.py`。主要設定都在 `config.py`：

| 設定 | 預設 | 意義 |
| --- | --- | --- |
| `ROLLOUT_EPISODES` | `32` | 一批 rollout 的完整 episodes 數；最後不足一批仍訓練。 |
| `PPO_MINIBATCH_SIZE` | `40` | 一次 Adam 更新使用的 transitions 數；最後不足一批仍更新。 |
| `PPO_EPOCHS` | `5` | 同一批 rollout 重複訓練幾輪，每輪可重新打亂。 |
| `PPO_CLIP_EPSILON` | `0.2` | PPO surrogate objective 的機率比 clipping 區間為 `[0.8, 1.2]`。 |
| `GAE_LAMBDA` | `0.95` | GAE 的時間加權係數；1 對應完整 episode 的 Monte Carlo return 減 value。 |
| `GAMMA` | `0.99` | 未來 reward 的折扣率。 |
| `LR` | `0.0001` | Adam learning rate。 |
| `PPO_ENTROPY_COEF` | `0.01` | 探索用的 entropy bonus 係數；0 關閉。 |
| `PPO_VALUE_CLIP_EPSILON` | `None` | 預設使用一般 value MSE；設定正數可開啟相對舊 value 的 clipping。 |
| `PPO_TARGET_KL` | `None` | 預設跑滿 epochs；設定正數時，mini-batch approximate KL 超過其 1.5 倍即停止後續更新。 |
| `PPO_SHUFFLE_MINIBATCHES` | `True` | 每個 epoch 重新打亂 transitions；False 可用於除錯。 |
| `PPO_ADAM_EPSILON` | `1e-5` | Adam 的數值穩定常數。 |
| `CRITIC_LOSS_WEIGHT` | `0.5` | Value MSE 的權重。 |
| `GRADIENT_CLIP_NORM` | `1.0` | 每個 PPO mini-batch 累積梯度後的 global norm 上限。 |
| `TRAIN_MICROBATCH_SIZE` | `2` | 每次前後向計算的 transitions 數，累積至 mini-batch 後更新；較小值節省 activation 記憶體。 |
| `NORMALIZE_ADVANTAGES` | `True` | 整批 rollout 的 actor advantage 正規化一次，各 epochs 共用。 |
| `VALIDATION_INTERVAL` | `100` | 到達此間隔後，在下一個完整更新邊界執行驗證；最後一定驗證。 |
| `VALIDATION_EPISODES` | `20` | 每次驗證使用完全相同的地圖、client 位置與 demand 情境。 |
| `VALIDATION_RANDOM_SEED` | `20261030` | 控制驗證地圖切分與固定 clients，獨立於訓練亂數。 |
| `VALIDATION_MAP_DIR` | `None` | 可指定獨立驗證地圖資料夾；不得與訓練地圖路徑重疊。 |
| `VALIDATION_MAP_FRACTION` | `0.2` | 未指定驗證資料夾時保留的地圖比例，向上取整，至少留一張訓練地圖。 |
| `BEST_MODEL_FILENAME` | `'best_model.keras'` | 此次訓練驗證分數最高的模型。 |

例如 5 顆 AP、`ROLLOUT_EPISODES = 32`，每批有 160 transitions。以每個 mini-batch 40 筆、5 epochs 計算，每批有 `160 / 40 × 5 = 20` 次 Adam 更新。`TRAIN_MICROBATCH_SIZE = 2` 表示每次只前後向計算 2 筆，累積 20 個小批梯度後做一次 Adam 更新。梯度以該 mini-batch 實際筆數平均，最後一批不足 40 筆也正確加權；地圖大小不同時分開做前後向計算。收集完整 rollout 期間權重固定，完成所有 PPO epochs 後才收集下一批資料。

取樣當下保存 masked action log probability、critic value、位置 mask 與 terminal 標記。每個 episode 獨立計算 GAE，放完最後一顆 AP 時 terminal bootstrap 為 0，不會串接下一個 episode。GAE advantage、lambda-return target 與舊 log probability 在所有 PPO epochs 中固定。Actor 使用 PPO-Clip 的 `min(ratio × advantage, clipped_ratio × advantage)`；critic 使用 value MSE，並加入 entropy bonus。總 loss 為 `actor_loss + CRITIC_LOSS_WEIGHT × critic_loss - PPO_ENTROPY_COEF × entropy`。

`PPO_CLIP_EPSILON` 限制 surrogate objective 中機率比的獎勵誘因，並不保證所有 action 的機率比都留在該區間；也不同於 `GRADIENT_CLIP_NORM`。GAE 改變的是訓練用的 advantage/target；reward 數值與計算方式、channel model、SNR、Mesh、OFDMA、CNN/GNN 架構均沿用原設定。實驗快照的 `TRAINING_ALGORITHM` 為 `ppo_clip_v1`。

驗證和一般 predict 相同，使用合法位置中的 greedy argmax；不進行梯度更新。以目前 10 張地圖、預設比例為例，會固定保留 2 張驗證、8 張訓練。只有一張地圖時，改成在該圖上使用固定 clients，並在輸出及紀錄中標示：此結果無法衡量未見地圖的泛化能力。驗證資料用來挑模型，正式論文的最終測試應另外保留測試情境。

每個驗證情境放完所有 AP 後計分，再取情境平均：

`score = COVERAGE_REWARD_WEIGHT × coverage + THROUGHPUT_REWARD_WEIGHT × mean_satisfaction + CONNECTIVITY_REWARD_WEIGHT × mesh_connectivity`

目前權重為 `0.2 / 0.7 / 0.1`。`mean_satisfaction` 是每位 client 的 `min(實際 throughput / demand, 1)` 平均；`mesh_connectivity` 是最大連通 AP 群的 AP 數／全部 AP 數。這裡評估最終部署成果；訓練每步使用增量 reward 與下述 True/False conn 定義。沒有 clients 時 throughput 項視為 0。分數相同時保留較早模型。

Connectivity reward 的第一顆 AP 視為 Mesh 起點，`conn=1`；後續 AP 連到放置前的 serving mesh 才是 1，未連入則為 0。這個 conn 仍乘上 `CONNECTIVITY_REWARD_WEIGHT`（目前 0.1），所以第一顆 AP 的 connectivity 加分為 0.1。訓練圖右下角為每個 episode 的 conn 平均：五顆 AP 都成功時，`[1,1,1,1,1]` 的平均為 1.0。它仍與最終 `mesh_connectivity` 不同；例如較早未連入的 AP 後來被其他 AP 橋接，最終 Mesh 可以完全連通，但較早的 conn 不會回溯加分。此規則適用新訓練與新 predict，既有訓練紀錄保持當時的實際 reward。

驗證及一般 checkpoint 都在更新完成後執行。例如 rollout 間隔 32、驗證間隔 100，驗證在 EP 128、224、320 等完整 rollout 訓練完成後進行，而最後 EP 2000 一定驗證。因此印出的最佳 episode 是「已訓練至該 episode 的已驗證模型」，不代表逐一驗證了每個 episode。

每次訓練產生獨立的 `model/AP_MESH_OFDMA_...` 資料夾，內含：

- `best_model.keras`：驗證分數提升時立即儲存，後續較差模型不會取代它。
- `model.keras`：最後一次更新後的模型。
- `config.json`：設定快照、最佳 episode、指標與檔案位置。
- `validation_scenarios.json`：訓練／驗證地圖清單、固定 client 位置與需求。
- `validation_history.json`、`.csv`、`.png`：各次驗證的最終部署指標與曲線。
- `best_validation_trials.json`：最佳模型的各驗證情境 AP 座標、連線與 client throughput。
- `update_history.json`：每個 rollout 的 mean_loss、policy_loss、value_loss、mean_episode_reward、coverage_ratio、throughput_satisfaction、episodes、transitions、PPO epochs、實際 optimizer 更新數、梯度 norm、entropy、approximate KL、clip fraction 與 explained variance。
- `ppo_minibatch_history.json`：每次實際 Adam 更新前的 mini-batch loss、policy_loss、value_loss、KL、entropy、clip fraction、explained variance、梯度 norm 與所屬 epoch；也附上該 rollout 的 mean_episode_reward、coverage_ratio、throughput_satisfaction。保留原有 actor_loss、critic_loss 欄位，分別等於 policy_loss、value_loss。
- `training_history.npz`、`.png`：依原 episode／placement 順序保存的資料及圖表。Loss、entropy、KL、clip fraction 是該 rollout 所有 PPO 更新後再評估的數值；value 是取樣時的舊 value，return_target 是固定 GAE lambda-return。Reward 與 coverage 等部署指標仍是取樣當時的觀測。
- `checkpoints/`：依 `CHECKPOINT_INTERVAL` 保存的一般中間模型。

JSON 中的 `policy_loss` 與 `value_loss` 分開保存，皆為乘上 loss 係數前的數值。`mean_episode_reward` 是此 rollout 各 episode「所有 AP 步驟 reward 的未折扣總和」再取平均；`coverage_ratio` 是各 episode 放完全部 AP 的 coverage 平均；`throughput_satisfaction` 是各 episode 最終的平均 client demand 達成比例再取平均。後兩項以 0～1 儲存，例如 0.8 表示 80%。這些部署指標來自訓練取樣，mini-batch JSON 中同一 rollout 的各筆會附上相同的三個數值；它們不是 PPO epoch 更新後重新部署的結果，也不是固定驗證集的指標。最後不足一個完整 rollout 時，以實際收集的 episodes 計算。固定驗證結果仍在 `validation_history.json`。

訓練完成後會印出最佳驗證 episode、score、coverage、throughput satisfaction、mesh 比例，以及最佳／最終模型的完整路徑。這些摘要會先印出，再顯示可選的訓練圖表視窗。

每個 rollout 的終端摘要包含 `EV=...`（Explained Variance）：`1 - Var(target - predicted_value) / Var(target)`。Mini-batch 的 EV 使用更新前的實際 critic 輸出；整個 rollout 的 EV 使用完成所有 PPO 更新後的 critic。Target 皆為該 rollout 固定、未正規化的 GAE lambda-return。EV 在整個 mini-batch／rollout 上共同計算，不平均各 microbatch 的 EV。Target 變異數為 0（包含單筆資料）時無法定義，JSON 記為 `null`、終端顯示 `EV=N/A`。EV 接近 1 表示能解釋 target 的變化，0 表示無法降低殘差變異，負值表示殘差變異大於 target 變異；EV 不衡量固定偏差，需搭配 critic MSE 查看。終端仍以每個 rollout 摘要輸出，mini-batch 的詳細數值依 epoch 存於 JSON。

當 `PREDICT_MODEL_PATH = None`，單次及 batch predict 會從最新完成且架構相容的訓練資料夾選取 `best_model.keras`。舊的已完成訓練沒有 best 模型時，仍使用其最終模型。可透過明確設定模型路徑選擇 final 或中間 checkpoint。現有模型不會被本次程式修改重新訓練或重新評選。

PPO 的實作參考 [Schulman 等人的 PPO 原始論文](https://arxiv.org/abs/1707.06347)。`PPO_TARGET_KL` 與 value clipping 是可選控制項，預設皆關閉。固定驗證與最佳模型挑選維持原本 objective 權重；更換訓練演算法不會自動保證部署成果提升，需以相同評估情境比較。
