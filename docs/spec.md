# 資料與標注規格

## 標注

第一版只標一種物件：`surfer`。

- bbox 框人物身體，不把整張衝浪板納入框內。
- 身體被浪、板或其他人物遮住時，依可推定的人體範圍標注。
- 無法可靠判斷為人物時不要建立 bbox。
- 動作模型取 crop 時會在 bbox 外多留背景（目前 1.55 倍），讓模型看到板與浪；
  這不改變 bbox 標注規則。

三項動作各自獨立，可同時成立：`chasing_wave`／追浪（`W`）、`takeoff`／起乘（`E`）、
`surfing`／衝浪（`R`）。就這三種，沒有其他類別；每個 bbox 至少要有一項成立。

各動作精確的開始與結束條件，待第一批影片試標後補上正反例。

## 部分標注

可以只框一張圖裡的部分 surfer，這些框仍可訓練動作分類。但該圖片預設不進 detector
dataset，避免未框的 surfer 被當成背景；只有勾選「區域內所有 surfer 都已框完」的圖片
才可用於 detector。

## 來源

- 接受公開 Google Drive 資料夾／單一影片，或 Facebook 貼文／影片連結。
- Drive 來源必須允許知道連結的任何人檢視及下載。
- Facebook 只保留清除 `__cft__`、`__tn__` 等追蹤參數後的 canonical URL；
  密碼、cookie、token、session 不可寫入 UI、設定檔、log 或 Dataset。
- 保留 Drive file ID、原始檔名、時間碼與 checksum，確保每張影格可追溯。
- 每個 Dataset 必須有 `sharer_name`、`source_provider`、canonical `source_url`、
  來源標題與影片數。無法從公開連結可靠取得分享者時由使用者手動填寫。

## 兩種訓練資料

- Detector：獨立 JPEG 與人物 bbox。
- Action：同一 `track_id` 的連續影格或短片段與動作標注。

## 切分

train/valid/test 預設 80/10/10、seed 42，先按來源影片分組再隨機切分
（`surf_track/dataset/split.py`）。同一片段的相鄰影格不可分散到不同 split。

## Drive 目錄慣例

```text
SurfTrackDatasets/
  sources/
  datasets/
    surftrack-v0001/
      frames/ clips/ annotations/ splits/ manifest.json
```

不要把所有影格放在同一個資料夾；依來源影片與資料版本分層。
