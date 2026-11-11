# AP-client SINR / AP-AP SNR 模型

環境版本：`ap_mesh_snr_access_sinr_ofdma_v2`。神經網路結構維持
`gnn_cnn_client_ap_attention_v4`，但輸入與服務環境的意義已改變，必須重新訓練。

## 兩種鏈路的用途

- **AP-AP**：仍用雙向參考 SNR 與 `MESH_SNR_THRESHOLD_DB = 4.0` 建立所有達標邊，
  允許環狀拓樸。最大連通分量提供服務，第一顆 AP 自成 Mesh。
- **AP-client**：依每個候選服務 AP 的 SINR 選擇單一 AP，且必須在服務 Mesh 中、
  SINR 不低於 `CLIENT_SINR_THRESHOLD_DB = 4.0`。沒有合法候選者時不關聯，throughput=0。
- Mesh 是無干擾參考條件下的潛在連通圖；不模擬 AP-AP 資料轉送或其資源開銷。

## 干擾假設與計算

`ACCESS_INTERFERENCE_MODEL = 'full_band_all_aps_always_on'`：所有已部署 AP
在相同參考頻寬持續同時同頻發射，包括沒有關聯 clients、以及不在服務 Mesh 的 AP。
干擾活動不隨被評估 clients 的 demand 是否達標而停止，屬固定背景滿載干擾近似。
不是逐包 Wi-Fi MAC、CSMA/CA、碰撞或重傳模擬。

對 client u、候選服務 AP i：

`SINR_ui = P_iu / (N_u + sum(P_ju for every deployed AP j != i))`

- 所有接收功率都在 **client 位置**取值；不是量在服務 AP 位置。
- dBm 先用 `10**(dBm/10)` 轉 mW，再加總；不取 AP 數量平均。
- 服務 AP 不作為自己的干擾；其他 AP 的干擾不由 Mesh 邊決定。
- 用每個候選者的其他功率直接加總，避免總功率減去強訊號的浮點消去誤差。
- 不同 client 各自具有不同接收功率與干擾。
- 全部同頻、固定功率、持續發射時，最高 SINR 與最高 RSSI 的排序相同，但關聯門檻、
  MCS 和 throughput 可能與舊 SNR 模型相差很大。

每新增一顆 AP，訓練、驗證、單次 predict、batch 的共用環境函式都會重新計算
**所有 clients 的所有候選 SINR、關聯、各 AP 的 RU 與 throughput**。
既有 AP 的傳播地圖可重用，不需要重新做相同射線計算。

## MCS / RU / throughput

維持現有 MCS threshold 表作為等效 SINR 接收性能近似，不宣稱是 IEEE 規範 PER 曲線。
MCS 由 SINR 決定，PHY rate 仍用 RU 資料子載波數 × 每符號位元數 × 編碼率 ×
空間串流數 / OFDM 符號時間計算；MAC efficiency、等大小 RU、超額 client 的輪流服務、
以及 `T = min(demand, scheduled_capacity)` 的公式不變。不能再額外除一次 client 數量。

本模型採平坦頻率響應、固定 PSD、全頻帶干擾近似，RU 分配後不依頻率位置重算 SINR；
未使用 RU 不產生干擾空白的收益，也不重新集中功率。RU 仍透過資料子載波數改變速率。

## 接收功率、coverage 與模型狀態

- 傳播函式保留低於 -100 dBm 的實際接收功率，避免把極弱訊號抬高成虛假干擾。
  距離、牆體損耗與射線演算法不變；AP-AP 環境入口保留舊 RSSI 範圍以保持 Mesh SNR 判斷。
- Coverage 仍是服務 Mesh 的 RSSI >= -60 dBm，在非牆體格點的比例。
  它是訊號覆蓋，不是 SINR 覆蓋。
- CNN 9 通道、節點 15 特徵與 attention 結構不變；client 節點第 5 欄改存
  正規化 SINR（-40～40 dB 映射至 0～1，範圍外裁切）。RSSI 仍使用原正規化。
- Reward 公式與權重、PPO 設定不變；throughput 增量會自然反映新增干擾造成的退步。

## 新報表欄位

- AP-client：`sinr_db`、`connected_AP_SINR`；舊 `snr_db` / `connected_AP_SNR`
  不再用於 client。
- 各候選 client link：RSSI、SINR、干擾 mW / dBm、雜訊 dBm、干擾加雜訊 dBm、
  干擾 AP 清單、Mesh 資格與是否達標。
- 各 client allocation / batch CSV：SINR、MCS、RU、throughput 與關聯鏈路的功率資料。
  未關聯時沒有服務鏈路，這些關聯功率欄位為 null；完整候選資訊仍在 `client_links`。
- 無其他干擾源時 `interference_mw=0`，其 dBm 欄位為 null，避免非標準 JSON 的負無限大。
- 圖上 AP-client 註記、右側欄、client 表格與 batch SINR ECDF 均使用 SINR。
  AP-AP 邊與 Mesh 表格仍標 SNR。

## 舊模型與結果

舊 SNR 模型與既有報表保留原檔。新環境不會自動載入舊模型，也不會將舊 client SNR
報表改標為 SINR。先執行新環境訓練後，再執行 predict / batch；新實驗會保存 SINR 設定快照。
原本 Gamma sweep 的比較屬於 SNR 環境，不能直接作為新環境的模型效果。
