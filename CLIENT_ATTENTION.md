# Client spatial attention：gnn_cnn_client_attention_v3

> 本文件是 attention 架構的歷史說明。目前 AP-client 品質特徵、關聯與速率已改為 SINR；
> AP-AP Mesh 仍使用 SNR。最新環境定義見 [SINR_MODEL.md](SINR_MODEL.md)。

目前 v4 保留本文件的 client 分支，另外加入 AP 空間分支；新架構與重新訓練方式見 `AP_ATTENTION.md`。

此版本維持原本多尺度 CNN、GNN message passing、PPO、SNR channel model、最強 SNR
單一 AP 關聯、OFDMA RU 分配，以及最大連通分量提供服務的定義。

## CNN 與 GNN 融合

Actor 除了原本的全域 graph 摘要，新增 `SpatialClientAttention`：

1. 保留 GNN 平均之前的逐節點特徵，串接原始節點特徵和該節點精確格點的 CNN 特徵。
2. 每個候選 AP 格點以 CNN 特徵和正規化座標形成 query；真實 client 節點形成 key/value。
3. 配對分數加入候選位置到 client 的相對 x/y、距離平方、剩餘需求和缺口比例。
4. 加權摘要同時保留相對幾何，避免只有一個 client 時 softmax 權重恆等於 1，
   導致所有位置得到相同 context。Actor 將此 16-channel context 與 CNN 和全域圖資訊融合。

已滿足的 client 仍保留節點與精確位置，因為後續 AP 可能改變其關聯和 RU 分配。
padding 與 AP 節點不作為 client attention 的 key；沒有 client 時 context 嚴格為零。
第一顆 AP 放置前也可讀取所有 client，無須增加不存在的無線連線。
Critic 繼續取得全域圖資訊與融合後的空間摘要。

Attention 預設 16 維，候選格點每 1024 格計算一次，避免建立格點對格點的注意力矩陣。
配對範圍是候選格點對節點，不是全圖格點兩兩連接；仍需要依實際地圖與 microbatch
大小評估記憶體。距離與需求特徵不是 SNR 或穿牆可達性的替代計算。

## 九個 CNN channels

| index | 意義 |
|---|---|
| 0 | 牆體材質 |
| 1 | 禁放區域 |
| 2 | 已放置 AP 的精確格點 |
| 3 | client 精確格點的剩餘 demand / throughput reference |
| 4 | 目前 serving mesh 的最大 RSSI |
| 5 | 剩餘 AP 放置比例 |
| 6 | 原始 client demand / throughput reference，保留完整需求 |
| 7 | 小尺度剩餘需求密度 |
| 8 | 大尺度剩餘需求密度 |

每步以實際重新計算的 allocation 建立 `remaining=max(original_demand-throughput,0)`。
已達標歸零；若下一次重新關聯或 RU 分配造成 throughput 下降，剩餘需求會重新出現。
這是狀態特徵；實際 demand、reward 達成率的分母仍是原始需求，沒有多 AP throughput 相加。

密度圖為每個 client 的剩餘 Mbps 乘 Gaussian 後相加，使用實際格點公尺尺寸。
`CLIENT_DENSITY_SIGMAS_METERS=(1.0,3.0)` 控制兩個尺度；
`CLIENT_DENSITY_REFERENCE_MBPS=100.0` 控制固定縮放 `density/(density+reference)`。
不以每張圖最大值正規化，所以需求減少不會又被拉回相同的最大值。
熱圖描述幾何上的需求集中程度，可跨牆延伸；真正連線與 throughput 仍由原物理模型計算。

## 十五個 GNN node features

原本 0..12 欄保持意義：AP/client 類型、正規化 x/y、RSSI、SNR、容量、原始 demand、
實際 throughput、達成比例、連線狀態、RU data share、time share。
新增 13：剩餘 Mbps / reference；14：剩餘需求 / 原始 demand。
AP 的新增欄位彙整其已關聯 clients 的缺口；client 顯示自己的缺口；零需求與 padding 為零。

## Reward 與模型相容性

依本次指定，`COVERAGE_GAIN_SCALE` 和 `THROUGHPUT_GAIN_SCALE` 都設為 `1.0`。
比例增量本來落在 [-1,1]，因此不再於 25% 增量時飽和。
權重 0.2/0.7/0.1、gamma=0.99、新 AP 當下加入 Mesh 的 0/1 獎勵和 PPO 計算維持原設定。
這不是 `gamma=1` 的 objective 增量獎勵；訓練與終局 objective 仍有中途連通和折扣差異。

新實驗快照會保存架構版本與需求熱圖參數。單次預測、批次預測和固定驗證均使用同一個
state builder；預測使用模型快照的特徵參數與 radio 格點尺寸，不受之後 live config 修改影響。
CNN-only/GNN-only 介面同步接受新輸入尺寸，其 backbone 保持原消融實作。

舊 v2 的 6-channel/13-feature 權重不相容，保留原檔。需要重新訓練 v3 模型，之後才能使用
單次或批次預測。自動選模只接受相同架構的已完成模型，明確指定舊模型也會提示重新訓練。
增加空間資訊路徑不保證訓練收斂或需求達標率提升，需比較固定場景的最終部署結果。

## 驗證結果

- `MPLBACKEND=Agg`、`CUPY_CACHE_DIR=.test_cupy_cache`，執行
  `python -m unittest discover -s tests -v`：64 項，63 通過、1 項可選專案不存在而略過。
- 包含剩餘需求 100/50/0、已滿足後重新出現缺口、零需求、固定公尺尺度熱圖、
  client permutation、padding/no-client 遮罩、單一 client 空間反應、梯度與 tf.function、
  儲存／重載、實際 PPO 訓練、驗證、單次／批次預測一致性。
- GPU：TF 2.10、RTX 4070，200×200 與 500×500 各完成一次兩 transitions、
  microbatch=2 的 PPO 更新；MAX_NODES=64、clients=10，attention 權重均有實際更新。
  TensorFlow allocator 峰值分別 382,341,632 bytes（約 0.36 GiB）與
  2,286,761,472 bytes（約 2.13 GiB）。這是合成小批次更新，不包含 RF 傳播、
  完整 rollout 的主記憶體儲存與其他程式，也不是完整訓練的顯存上限。
- SHA-256 確認 `config_radio.py`、物理環境、傳播、SNR、容量、RU 分配、reward 函式
  與 PPO 實作保持原內容；reward 改變來自 config 的兩個 scale。
- 原始記錄：`diagnostics/client_attention_v3/tests.log`、`gpu_check.json`、
  `unchanged_before.json`、`unchanged_after.json`。GPU 重現程式：
  `diagnostics/check_client_attention_gpu.py`。
