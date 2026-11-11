# 多尺度 CNN：gnn_cnn_multiscale_v2

> 以下為 v2 歷史架構說明。目前主模型已升級為 `gnn_cnn_client_attention_v3`，
> 包含 9-channel/15-feature 剩餘需求特徵和逐格 client attention，詳見 [CLIENT_ATTENTION.md](CLIENT_ATTENTION.md)。

本次調整 `GNN_CNN_Network` 的 CNN 分支，維持六個地圖 channel、13 個節點特徵、
GNN message passing、全域圖摘要、reward 與環境定義。
CNN-only / GNN-only 的獨立消融模型尚未改為此多尺度結構。

## 取得大範圍資訊

CNN 使用四次 `same` max pooling，解析度為原圖、1/2、1/4、1/8、約 1/16。
高解析度分支通道數為 16、24、40、64；96-channel 的主要情境運算在 1/16 執行。

低解析度 bottleneck 使用 dilation 1、2、4、8 的卷積。
依卷積與 pooling 計算，此路徑到 bottleneck 的理論感受野為 556 格，尚未計入 decoder。
另外使用 1×1、2×2、4×4 adaptive average pooling，彙整全圖與區域分布，再融合到 bottleneck。
因此任何尺寸地圖都具備取得全圖摘要的路徑，而非只依賴有限大小的局部卷積視窗。
全圖摘要仍是壓縮資訊，不等於保留每個遠方 client 的完整細節。

## 保留精確位置

每層 decoder 都依對應 encoder 的實際尺寸 resize，並接回同解析度的 skip features。
最後一層還直接接回原始六個輸入 channel，保留薄牆、單格 client demand、AP 位置等細節。
不對 client 座標做取整到粗網格的處理，也不把最終 action 網格縮小。
奇數尺寸、非方形甚至 1×1 地圖都輸出原尺寸的每格位置機率。

全解析度 CNN 輸出 16 channels，GNN 的全域摘要投影為 16 channels 後沿用原本的 broadcast + concat。
因此高解析度融合張量由舊版的 256 channels 降為 32，後接 32-channel 卷積。
GNN 節點處理和平均彙整的數學定義維持不變，尚未加入位置相關 cross-attention。

## 訓練與預測

```powershell
python GNN_CNN_DRL_train.py
# 完成新模型訓練後：
python GNN_CNN_DRL_predict.py
python GNN_CNN_DRL_predict_batch.py
```

`config.MODEL_ARCHITECTURE` 現在是 `gnn_cnn_multiscale_v2`。
新實驗的 config.json 會記錄架構版本，單次與批次預測只載入相容架構。
舊模型保留原檔；不能直接載入新架構，需要重新訓練。
新模型尚未完成時，預測會明確提示先訓練，不會自動誤用舊權重。
如設定了 `PREDICT_MODEL_PATH` / 批次的 `MODEL_PATH`，也需指定新版模型。

此修改提供較大視野與精細定位的學習能力，不保證訓練後 AP 一定遠離角落。
仍應以固定測試情境、集中不同區域的 clients、最終需求達標率和 coverage 驗證效果。
CNN-only 消融比較若要隔離 GNN 的貢獻，之後應讓其 CNN backbone 與此版本一致。

## 本次驗證

- `python -m unittest discover -s tests -v`：32 項通過；1 項因舊版獨立 PPO 專案不在工作目錄而略過。
- 包含奇數/非方形/1×1 輸出尺寸、遠方 client 到候選位置特徵的梯度路徑、
  相鄰單格 client 可區分性、完整模型與權重重載、實際訓練更新，以及單次/批次預測。
- RTX 4070、TF 2.10，合成 `10×500×500×6` 軌跡完成一次真實 actor/critic GPU 更新，
  loss 與更新後權重均有限；TensorFlow 回報峰值 5,041,927,680 bytes（約 4.70 GiB）。
  這項檢查不包含 RSSI 傳播或其他程式的顯存使用，並非所有訓練配置的顯存上限。
- 原始 GPU 檢查結果：`diagnostics/multiscale_gpu_check_c50c9ddc.json`。
