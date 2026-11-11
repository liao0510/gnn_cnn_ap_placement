# AP spatial attention v4

> attention 結構仍是 v4；AP-client 的環境與品質特徵已由 SNR 改為 SINR，需要重新訓練。
> 以下保留架構修改當時的紀錄，最新模型假設見 [SINR_MODEL.md](SINR_MODEL.md)。

Architecture ID: `gnn_cnn_client_ap_attention_v4`.

本次保留 `J = 0.2 coverage + 0.7 satisfaction + 0.1 mesh` 的設定、原 reward 計算、通道模型、最大 SNR 關聯及等尺寸 RU 分配。新增的是模型讀取既有 AP 的方式，沒有新增距離懲罰、最小間距或 demand 必須全部達標的限制。

## 輸入與融合

CNN 仍有 9 個通道、每個 GNN 節點仍有 15 個特徵。精確 AP 格點圖仍保留。AP 節點原本未使用的 RU/time share 欄位（11/12，從零開始）現在填入環境實際配置；沒有改動環境配置結果。各 AP 的關聯 client 數由 GNN adjacency 與真實 client mask 計算，不依賴 demand 推估。

新增 `training_method/AP_attention.py::SpatialAPAttention`，將每個候選格點直接對齊每顆有效 AP，使用：

- 原始 AP 精確座標、GNN 節點嵌入及 AP 所在格點的 CNN 特徵。
- 候選點與 AP 的相對 x/y、距離平方與兩個局部範圍的接近程度。
- AP 的關聯 client 數、需求量、提供的 throughput、平均達成率、Mesh 狀態、RU/time share 及剩餘需求。

AP 分支輸出 16 個可學習的 context 通道，以及 5 個直接保留、不經 attention 加權平均或 learned projection 的空間摘要：

1. 到最近 AP 的正規化座標距離平方，除以 2，落在 0～1。
2. 到最近 AP 的小範圍 Gaussian proximity。
3. 小範圍 AP 密度：各 AP Gaussian 相加後，以 `density / (1 + density)` 正規化。
4. 大範圍 AP 密度：使用相同的固定正規化方式。
5. AP 數量 `n / (1 + n)`。

預設 Gaussian 標準差為 5／20 格；目前每格 0.1 m，對應 0.5／2 m。密度相加可以分辨「附近只有一顆 AP」與「多顆 AP 集中」，不會因為除以每張圖最大值而消除數量差異。這些通道是觀測特徵，沒有直接扣分或禁止放置。

融合為 `CNN(16) + global GNN(16) + client attention(16) + AP attention(21)`，共 69 通道，再經 32 通道卷積送入 actor／critic。AP 數量為零時 AP 分支完全為零；第一顆 AP 仍由 client、地圖及全域圖資訊決定。

AP 節點會先打包，排除 clients 與 padding，再分 1024 個候選格點一批計算，避免建立整張地圖格點對格點的 attention 張量。

## 重複位置與附近集中

訓練取樣、單次 predict、batch predict 與驗證均沿用共用 `placement_mask`，排除已放置 AP、clients、牆體及禁止區。相同格點的重複放置已由此遮罩保證排除。

`(197,2)、(195,2)、(196,2)` 是相鄰格點，不是相同格點。v4 提供更直接的距離／密度與負載資訊，讓模型學習這種集中是否有新增效益；attention 本身不保證相鄰集中完全消失。在 clients 密集、需要分攤 RU 時，附近多放 AP 仍可能合理。

## 訓練與預測

請重新執行 `python GNN_CNN_DRL_train.py` 訓練 v4。舊 v3 權重、紀錄及圖片保留；架構檢查會拒絕將它們當作 v4 載入。`config.py` 中的 `PREDICT_MODEL_PATH` 已設為 `None`，自動選取最新完成、架構相容的模型之 best checkpoint；在第一個 v4 訓練完成前，predict 會提示尚無相容模型。

本次沒有自動開始完整訓練，也不能把未訓練 v4 的位置敏感度當成部署品質已改善。應使用新的 client 情境及相同物理設定，比較 v3／v4 的加權 objective、coverage、satisfaction、完全連通率與 AP 過度集中的案例比例。
