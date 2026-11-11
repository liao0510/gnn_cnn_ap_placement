# 批次預測與隨機 AP 對照

目前 client 關聯、MCS、throughput 與 client 圖表使用 SINR，AP-AP Mesh 仍使用 SNR。
需先訓練新環境模型；原有 SNR 結果保留。假設與欄位見 [SINR_MODEL.md](SINR_MODEL.md)。

在此專案目錄執行：

```powershell
python GNN_CNN_DRL_predict_batch.py
```

設定集中在 `GNN_CNN_DRL_predict_batch.py` 最上方：

| 設定 | 預設 | 用途 |
|---|---|---|
| `NUM_PREDICTIONS` | `100` | 成對評估次數；每次跑 GNN＋CNN 和 Random AP 各一次 |
| `RANDOM_MAP_EACH_RUN` | `False` | 固定 `MAP_PATH`，每次重新抽 client |
| `MAP_PATH` | `maps_200_200/map_0000.json` | 固定模式的地圖 |
| `MAP_DIRECTORY` | `maps_200_200` | 換圖模式隨機選取的資料夾，至少兩張 JSON |
| `MODEL_PATH` | `None` | 沿用 config 指定模型；未指定時選最新完成訓練的 best model（舊訓練無 best 時用 final），整批只載入一次 |
| `AP_COUNT` / `CLIENT_COUNT` | `5` / `10` | 兩組相同的 AP 與 client 數量 |
| `CLIENT_DEMAND_MBPS` | `100.0` | 每個 client 的需求 |
| `CLIENT_DEMANDS` | `None` | 可改為需求列表，長度需等於 client 數 |
| `RANDOM_SEED` | `42` | 整批可重現，每個 trial 持續抽新 client，不會每次重設種子 |
| `BOOTSTRAP_SAMPLES` | `2000` | 平均值及成對差異的 bootstrap 重抽樣次數 |
| `SHOW_PLOTS` | `False` | 是否開視窗；無論設定為何都輸出 PNG、PDF |

換圖模式避免連續選到同一張地圖，但允許隔幾次後再次選到同一張。
抽樣可能偶然產生相同 client 配置；不刻意排除，維持原本均勻抽樣。

## 公平比較方式

每個 trial 的兩組共用同一張地圖、同一組 client 座標、相同 demand、AP 數量、
模型訓練快照的無線參數與 OFDMA 配置。
GNN＋CNN 使用現有單次 predict 的 sequential masked argmax。
Random AP 是均勻隨機位置基準，不需要訓練；排除牆、禁止區、client 和已放 AP，
不額外要求接入 Mesh。兩組最後都用相同最大連通群規則服務 client。
此批次程式一律隨機產生 client，不使用 config 的手動 `PREDICT_CLIENT_POSITIONS`。
地圖、client、隨機 AP、統計分析使用獨立 RNG，分析抽樣不影響部署結果。

原本單次 predict、訓練程式、reward 和使用者目前的設定不會被批次程式改寫。
每次執行建立獨立時間戳資料夾，避免覆蓋先前結果。

## 圖表與資料

所有指標都取放完全部 AP 的最終值，不平均中間部署步驟。

| 檔名（圖另有 PNG、PDF） | 內容 |
|---|---|
| `per_trial_metrics` | 每次 coverage、平均 throughput satisfaction、Mesh 連通比例 |
| `metric_distributions` | 上述三指標的箱型圖與 ECDF |
| `summary_metrics` | 九項指標平均值與 95% bootstrap CI |
| `paired_differences` | 每次 GNN－Random 的三項差異、勝率、平手率及差異平均的 CI |
| `client_distributions` | client 實際 throughput、需求達成比例、已連線 SNR 的 ECDF |
| `trial_metrics.csv` | 每次、每種方法的最終量化指標，可用 Excel 開啟 |
| `client_metrics.csv` | 每個 client 的連線 AP、SNR、RU、throughput、demand、達標狀態等；有 client 才產生 |
| `summary.csv` / `summary.json` | 平均值、樣本標準差、中位數、CI、成對差異等 |
| `trials.jsonl` | 每完成一次立即寫入：地圖、client/AP 座標、Mesh 及分配詳細結果 |
| `evaluation_config.json` | 實際模型路徑、訓練/無線快照、種子、候選地圖與指標定義 |

指標定義：

- **Coverage**：服務 Mesh 覆蓋的非牆格子／全部非牆格子，與訓練一致。
- **Mean satisfaction**：各 client 的 `min(throughput / demand, 1)` 平均；零需求視為達標。
- **Clients meeting demand**：實際 throughput 完全滿足 demand 的 client 比例。
- **Mesh connectivity**：最大 AP 連通群大小／全部 AP 數，不是 reward 的單步 `conn`。
- **Fully connected mesh**：單次為 0 或 1；跨次平均就是完全連通成功率。
- **Total throughput**：所有 client 實際交付 throughput 的總和，已套用 demand cap。
- **5th-percentile throughput**：每次內 client throughput 的第 5 百分位，再跨次統計；包含未連線的 0。
- **Jain fairness**：對正需求 client 的需求達成比例計算 `(sum x)^2 / (n * sum x^2)`。
  全部為 0 時定義為 0；公平性高不代表需求已達標，需搭配 satisfaction 看。
- **Client association rate**：已連線 client／全部 client。

零 client 或沒有正需求 client 時，相關無法定義指標記為空值，不強制填 100%。
所有比例在 CSV/JSON 以 0–1 儲存，圖上以百分比顯示。
成對比例差異以「百分點」解讀，例如 0.10 = +10 個百分點。

## 統計解讀

bootstrap 以整個 trial 為單位；先計算每次的 GNN－Random 差異再重抽樣，
不把同一 trial 內的 clients 當成獨立樣本來膨脹樣本數。
只跑一次時不提供 CI/樣本標準差。95% CI 為 percentile bootstrap 的估計，
小樣本或全為相同值時應審慎解讀；不自動宣稱統計顯著。

這些 CI 描述這一個已訓練模型在所抽評估情境上的不確定性，
不包含重新訓練不同種子的差異；固定地圖也不代表能推論所有地圖。
正式論文應使用未參與訓練的測試地圖，並以多個訓練種子另外評估。

參考：[Jain 公平性原始報告](https://www.cs.wustl.edu/~jain/papers/ftp/fairness.pdf)、
[NIST bootstrap 與 paired groups](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/bootplot.htm)。
