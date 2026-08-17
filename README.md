# SurfTrack

個人專案：建立可部署到 Jetson Orin Nano 8GB 的 surfer 即時辨識模型，並提供從
收集影片 → 標注 → 訓練的單頁本機 UI。

固定範圍：單路相機、10 Hz；找出畫面中所有 surfer，bbox 只框人物身體；每個 bbox 輸出
「追浪／起乘／衝浪」三項獨立機率。不負責選擇跟拍目標、控制相機或辨識身分。

## 快速開始

只需要 Python 3.11，沒有 Web framework（stdlib `http.server` + 原生 JS）：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .          # 加 '.[train]' 才會裝 torch，訓練才需要
./surftrack.py                      # http://127.0.0.1:8080
.venv/bin/pytest -q
```

抽影格需要 `ffmpeg` 在 PATH 上。

## 使用流程

1. **Data**：貼上公開 Google Drive 或 Facebook 貼文連結 → 掃描 → 建立 Dataset
   （必填分享者名稱）→ Ingest 下載影片並以 1 fps 抽影格。
2. **Label**：拉框、`W`/`E`/`R` 切換三種動作、空白鍵存檔跳下一張。
3. **Train**：勾選分享者，所有勾選的資料集合併成一個 MobileNetV3-Small 動作分類器，
   模型寫到 `var/models/action/`。訓練只在遠端 CUDA 主機執行（UI 需填 `user@host`），
   影格留在本機，只把 crop 逐 batch 串過去；checkpoint 串回本機。

同樣的訓練也可以從 CLI 跑，`--sharer` 可重複：

```bash
python -m surf_track.training.stream feed --sharer 分享者名稱 --host user@192.168.x.x
```

## Orin 部署契約

### 動作分類器（SGIE）

寫在這裡是因為兩者都很容易在部署時踩到，而且事後很難察覺：

- **SGIE 前必須用 pad probe 把偵測框擴成 `max(w, h) * 1.55` 的正方形**，不要用 nvinfer
  的非等比縮放（`maintain-aspect-ratio=0` 是預設值）。訓練時的 crop 是正方形，身體比例
  沒有被扭曲；而 90% 的框長寬比落在 [0.8, 1.25] 之外，直接拉伸會把「趴著／站立」這條
  最強的線索抹平。順帶也解決 nvinfer 的 16 px 輸入下限。
- **nvinfer 內建的 classifier parser 會對輸出取 argmax 且不套 sigmoid**，會把三個 logit
  折成一個互斥標籤，違背「三項獨立機率」。要用 `output-tensor-meta=1` 自己算 sigmoid。
- ONNX 要 export dynamic batch（一次 batch 20–30 個 crop，而不是逐一呼叫）。

### Detector（PGIE）

**匯出必須截掉後處理。** torchvision 把分數過濾與 NMS 烘進 forward，匯出的圖會帶
`If` / `NonZero` / `TopK`，TensorRT 的靜態 engine 吃不下。只匯出 `backbone` + `head`，把
decode 與 NMS 留給 DeepStream parser，圖就完全乾淨。訓練與本機評估仍用完整模型，Python
後處理沒問題，所以要自己寫 decode 的地方只有 parser 一處。

Faster R-CNN 系列不能用：`RoIAlign` 在網路中間而不是後處理，切不掉。
選 FCOS 而非 RetinaNet 是因為 anchor-free —— RetinaNet 的 parser 必須把 anchor 尺寸、
比例、每層 stride 一模一樣複製一遍，錯了不會報錯，只會讓 mAP 莫名偏低。

架構：`mobilenet_v3_large(norm_layer=FrozenBatchNorm2d)` 的 features，用公開的
`BackboneWithFPN` 取 index [4, 7, 16]（stride 8 / 16 / 32、通道 [40, 80, 960]），FPN 128
通道，`LastLevelP6P7` 補到 5 層，`FCOSHead(128, ..., num_convs=4)`。單類所以
`num_classes=1`，sigmoid head 沒有背景通道，**ground-truth label 必須全是 0 而不是 1**。
實測 0 個 live `BatchNorm2d` / 46 個 `FrozenBatchNorm2d` / 8 個 `GroupNorm`，所以 6GB 卡的
小 batch 不會毀掉 BN 統計。匯出實測 1246 nodes / 19 ops，CLEAN。

**三個會靜默壞掉的坑：**

1. `_mobilenet_extractor` 的預設 `returned_layers` 最細 stride 是 **32**，整張圖只有
   **540 個位置** —— 對中位數 87 px 的框毫無意義。必須明確指定 stride 8 起跳。
2. torchvision 的 `min_size=800, max_size=1333` 預設會**放大**你的輸入（實測 576×1024 →
   768×1344，算力多 2.3 倍且無提示）。必須明確傳 `min_size` / `max_size`。
3. stride 是 `padded_dim // grid_dim`，**不是** 2 的次方。只有兩軸都能被 128 整除才乾淨：
   1024×576 的 P7 是 stride_y=115（兩軸不等），1280×736 是 61 / 122。parser 會算錯。

**算力在 head，不在 backbone**（實測 77%），所以槓桿是 FPN 通道與 head conv 數：
896×512 下 fpn128/head2conv 9.16、fpn128/head4conv 14.79、fpn256/head4conv 52.98 GMAC。

輸入定 **896×512**（stride 乾淨、追浪的框還撐得住）。影格是 16:9（1168 張 1920×1080、
442 張 3840×2160），letterbox 進正方形會浪費 40% 像素在灰邊上且框大小完全不變：

| 輸入 | GMAC | 位置數 | 中位框高 | < 32 px | **追浪**中位 | stride |
|---|---|---|---|---|---|---|
| 640×384 | 7.93 | 5115 | 62 px | 213（13.2%） | 31 px | 乾淨 |
| **896×512** | **14.79** | **9548** | **87 px** | **77（4.8%）** | **43 px** | **乾淨** |
| 1024×576 | 19.02 | 12280 | 100 px | 51（3.2%） | 50 px | ✗ P7 115 |
| 1152×640 | ~23 | 15345 | 111 px | 34（2.1%） | 54 px | 乾淨 |

`chasing_wave` 的框是三類中最小的（浪把人推向鏡頭，站起來才變大），而它正是必須先開火
才能讓 tracker 建立 ID 的那一類 —— 所以解析度不是次要議題。640×384 下追浪只有 31 px。

框高是把每張影格按 `min(W/iw, H/ih)` 縮放後量的；原始框高中位數 200 px。

**上面沒有任何 ms 數字，因為還沒有在 Orin 上量過。** 要定案就在板子上跑
`trtexec --fp16 --shapes=images:1x3x512x896`，不要拿別人的 benchmark 外推。

## 專案結構

```text
surftrack.py            啟動器
surf_track/
  server.py             stdlib HTTP server 與 /api/v1 路由
  store.py              SQLite：datasets / images / annotations / jobs / training_runs
                        （training_runs 以分享者為範圍，不綁單一 dataset）
  ingest.py             下載影片、ffmpeg 抽影格的背景 worker
  config.py             Settings（環境變數 SURF_TRACK_*）
  drive/ facebook/      來源連結解析與下載
  dataset/split.py      依來源影片分組的 80/10/10 切分
  training/             action.py 訓練、manager.py 背景工作、stream.py 遠端 GPU
  web/                  index.html / app.js / styles.css
var/                    執行期資料（DB、影片、影格、模型、憑證）；不進 git
```

## 憑證

- 公開 Drive 走 `gdown`，不需要帳號或 API key。
- 私人 Drive 預留 `var/secrets/rclone.conf` + `~/.config/surftrack/rclone-config.pass`，
  目錄權限 700，尚未實際接上。
- Facebook 只保留清除追蹤參數後的 canonical URL；密碼、cookie、token 不得寫入 UI、
  設定檔、log 或 Dataset。

## 目前狀態

已完成：Drive 掃描與 ingest、標注 UI 與 SQLite checkpoint（意外關機的工作會標為可恢復）、
動作模型訓練與預覽、遠端 GPU streaming 訓練。

未完成：私人 Drive OAuth、Facebook 登入與影片下載、bbox detector 訓練與 Orin 部署。

規格見 [docs/spec.md](docs/spec.md)，開發約定見 [CLAUDE.md](CLAUDE.md)。

## 致謝

感謝以下提供影片用於模型訓練：

- [黃偉豪](https://www.instagram.com/mybaby04094911/)
- [布魯托衝浪攝影](https://www.instagram.com/brutalsurfcam/)
