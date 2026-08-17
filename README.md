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

Detector 吃的是「整張標完」的影格，pixels 會先降到 896×512 存進 `var/cache/frames/`，
模型寫到 `var/models/detector/`。Train 會在動作模型跑完後接著訓練它（整張標完的影格
不足 12 train / 3 valid 時自動跳過），也可以單獨從 CLI 跑：

```bash
python -m surf_track.training.stream feed-detector --host user@192.168.x.x
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

**匯出必須截掉後處理**，但理由**不是** TensorRT 吃不下 —— TensorRT 10.16 的 operator 表
把 `NonMaxSuppression` / `NonZero` / `If` / `TopK` / `RoiAlign` 全列為支援。真正的理由有三個：

1. **綁死的是 nvinfer 不是 TensorRT。** data-dependent shape 的輸出回報 `-1` 維度、要求呼叫端
   註冊 `IOutputAllocator`，而 nvdsinfer 交給 custom parser 的是它自己依 init 時已知維度預先
   配置好的 buffer。這兩個契約不相容。（**DS 9.1 上未經驗證**，NVIDIA 兩邊都沒明說。）
2. **不管怎樣都要寫 custom C++ parser。** nvinfer 對任何「NMS 在圖內」的輸出格式都沒有內建
   parser —— `nvdsinfer_yolo*_efficient_nms` 整個生態就是為此存在的。截掉後處理只要多寫
   FCOS 的 decode（anchor-free，約 25 行），沒有多付任何代價。
3. **DDS 會強迫同步執行**，直接打在 pipeline 的 10 Hz 預算上。

**PGIE 的縮放要設成 letterbox**：`maintain-aspect-ratio=1` 且 `symmetric-padding=0`。訓練時
torchvision 的 `GeneralizedRCNNTransform` 是等比縮放後把 padding 補在右下（1920×1080 → 896×504
再補 8 列），而 nvinfer 預設 `maintain-aspect-ratio=0` 會直接拉伸成 896×512。差異看起來很小，
但那是一個訓練時完全沒出現過的幾何，而且不會有任何錯誤訊息。

**不要規劃 `EfficientNMS_TRT`**：它在 TensorRT 10.12 被 deprecated，`EfficientNMS_ONNX_TRT`
在 **10.16 被移除**，官方替代是 `INMSLayer`。

訓練與本機評估仍用完整模型，Python 後處理沒問題，所以要自己寫 decode 的地方只有 parser 一處。

Faster R-CNN 系列不能用：`RoIAlign` 在網路中間而不是後處理，切不掉。
選 FCOS 而非 RetinaNet 是因為 anchor-free —— RetinaNet 的 parser 必須把 anchor 尺寸、
比例、每層 stride 一模一樣複製一遍，錯了不會報錯，只會讓 mAP 莫名偏低。

架構：`mobilenet_v3_large(norm_layer=FrozenBatchNorm2d)` 的 features，用公開的
`BackboneWithFPN` 取 index [4, 7, 16]（stride 8 / 16 / 32、通道 [40, 80, 960]），FPN 128
通道，`LastLevelP6P7` 補到 5 層，`FCOSHead(128, ..., num_convs=4)`。單類所以
`num_classes=1`，sigmoid head 沒有背景通道，**ground-truth label 必須全是 0 而不是 1**。
實測 0 個 live `BatchNorm2d` / 46 個 `FrozenBatchNorm2d` / 8 個 `GroupNorm`，所以 6GB 卡的
小 batch 不會毀掉 BN 統計。

**detector 必須以固定 batch 1 匯出，絕對不要加 dynamic axis。** 實測加了之後三個輸出全部變成
symbolic（`['Concat1790_dim_0', 'Concat1790_dim_1', 1]`）—— **9548 這個數字被摧毀**，而 nvinfer
要求輸出 shape 全靜態。靜態版是 **1062 nodes / 18 op types**，輸出 `[1,9548,1]` / `[1,9548,4]`
/ `[1,9548,1]`。動作分類器**相反**，它需要 dynamic batch（一次 20–30 個 crop）。

**decode 要烘進 ONNX，parser 裡不留任何幾何。** custom C++ parser 無法避免（nvinfer 對
`boxes`+`scores` 佈局沒有內建 parser），但把 FCOS 的 decode 放進圖裡，就能把 stride 表、FPN
層序、clamp 邊界這三個無聲坑從 C++ 搬到 Python，用 pytest 斷言。已驗證：以常數 `centers` /
`sizes` 算 `scores = sqrt(sigmoid(cls) × sigmoid(ctrness))`、`boxes = (cx − l·s, …)` 再 clamp，
對 torchvision 的 `box_coder.decode` + `clip_boxes_to_image`，9548 個位置的 max abs diff
**恰好 0.0**。烘進去後圖仍乾淨：1085 nodes / 23 op types、輸出全靜態。

parser 於是只剩約 40 行：**按 `layerName` 查 layer（不要按 index）**、用
`detectionParams.perClassPreclusterThreshold[0]` 過濾、把角點填進 `NvDsInferObjectDetectionInfo`
（**逐欄位指名賦值，不要 positional brace-init** —— 該 struct 在 DS 9.1 的欄位數沒有可靠來源）。

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

### DeepStream 9.1 / JetPack 7.2.1 上會靜默壞掉的事

版本：**DeepStream 9.1 對應 JetPack 7.2（L4T r39.2）**，TensorRT 10.16.2。DS 9.1 重建了所有
engine，所以 engine、custom parser、plugin 都要對 DS 9.1 重編，**絕對不要沿用 JetPack 6.x 的**。

**正規化的算術**（最容易無聲搞錯的一處）。nvinfer 的公式是
`y = net-scale-factor × (x − offsets)`，`x` 是 **0–255 的整數**、`offsets` 也在 0–255 單位並
在縮放**之前**相減。⚠️ NVIDIA 自家的 agent-skill 參考文件寫成 `(input × scale) − offsets` 且
說 offsets 是「mean / std」—— 照那個寫會得到 ~2.1 而不是 ~123.7，**差 58 倍、不會報錯、看起來
只像是「模型沒訓練好」**。以 plugin 手冊為準。

而且 `net-scale-factor` 是**單一純量**，所以 ImageNet 的 per-channel std（0.229/0.224/0.225）
**無法精確表示**。解法是把正規化烘進 ONNX（輸入端一個 Mul + Add，TensorRT 會把常數摺進第一層
conv，零成本），nvinfer 設 `net-scale-factor: 1.0` / `offsets: 0;0;0`。四個容易寫反的數字變成
一個開關。

`model-color-format: 0`（RGB）—— torchvision 的 transform 吃 RGB PIL，設成 1 會 R/B 對調且
不報錯。`offsets` 的順序跟著網路的 channel 順序，RGB 就是 `123.675;116.280;103.530`，不是你
從大多數範例 config 抄來的 BGR 順序。

**輸入幾何必須在訓練前決定。** torchvision 的 `GeneralizedRCNNTransform` 不會給你固定尺寸：
`min_size=max_size=640` 會把 1920×1080 變成 640×360 再 pad 成 640×384，那沒辦法交給固定形狀的
engine。選 896×512 的附帶好處是它的 aspect 1.75 對上來源 1.778 只差 1.6%，所以可以直接非等比
squash、`maintain-aspect-ratio=0`，**完全不需要 letterbox 座標還原**（那是「框整體偏移 50 px」
這類 bug 的長年來源）。SGIE 不受影響 —— nvinfer 是從 streammux 的全解析度 surface 裁切物件，
不是從 PGIE 的張量。

**`pyds` 已經 deprecated**，DS 9.0 起不再釋出 wheel，官方替代是 SDK 內附的 `pyservicemaker`。
一個實務限制：probe **可以改寫** `obj.rect_params`（所以 1.55 倍正方形 padding 可行），但
**無法刪除 object meta**（NVIDIA：「目前 pyservicemaker 沒有提供移除 object meta 的介面」）。

其他容易踩的：`pre-cluster-threshold` / `nms-iou-threshold` / `topk` 放在 `class-attrs-all`
而不是 `property`；ONNX 有任何 dynamic axis 就**必須**設 `infer-dims`；一定要設
`model-engine-file`，否則每次啟動都重建 engine（Orin Nano 上要好幾分鐘）。

**最危險的一個未知：nvinfer 的 per-object classifier cache。** 如果 SGIE 的結果只在每個 track
的**第一幀**出現，那麼一個在追浪階段第一次被看到的 track 會**一輩子回報追浪**，起乘的瞬間永遠
不會被觀測到 —— 「選最早起乘的人」這個核心情境直接死亡，**而且每個框、每個 ID 看起來都完美**。
上板第一件事就是這個 canary：300 幀內數「物件數」對「當幀取得新 tensor meta 的物件數」，穩態
必須相等。`network-type=100` 應該能繞掉（cache 分支掛在 classifier instance 上）。

**還有一個只能上板才知道的**：pyservicemaker 讀不讀得到掛在 object 上的 tensor meta
（`obj_user_meta_list`）。讀不到的話 fallback 是在同一個 `.so` 裡多寫約 20 行 classifier
parser，讓三個 sigmoid 以三個獨立的 `NvDsClassifierMeta` attribute 送出 —— 仍然是三項獨立
機率、仍然沒有 argmax，只是換運輸方式。

**detector 訓練好之前就能驗證整條 pipeline**：匯出一個 synthetic engine —— 把 cls/ctrness head
最後一層 bias 設成 `logit(0.9)`、regression bias 設成 1.5，再乘一個只在 N 個位置為 1 的常數
mask。得到每幀恰好 N 個框、位置已知的確定性假 detector，跑的是真 config、真 parser、真 engine，
所以 tracker + SGIE + probe 全部驗得到。**random-init 沒用**：`bbox_regression` 過 relu 後幾乎
全 0，框全退化，PGIE 一個物件都不會產生。檔名要帶 `_synthetic`，且不要從它讀延遲數字。

**GO/NO-GO 判準**：`trtexec` 量出 detector 單獨 **> 55 ms 就是 896×512 不成立**（100 ms 裡還要
塞 decode、NvDCF 20 目標、24-crop SGIE、兩個 probe，而且要跟 DeepStream 共用板子）。替代方案
README 已標好價：head 2conv 9.16 GMAC、640×384 7.93 GMAC —— 640×384 最後才用，它把追浪的中位
框從 43 px 壓到 31 px。

**Orin Nano 沒有 DLA**，GPU 是唯一的推論引擎。DS 9.1 的官方效能頁只有 AGX Thor / AGX Orin /
DGX Spark / dGPU，**Orin Nano 完全沒有已發表數字**。

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
