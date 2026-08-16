# Dataset 原則

## 來源

- UI 接受公開 Google Drive 資料夾或單一影片連結。
- 來源必須允許知道連結的任何人檢視及下載。
- UI 也接受 Facebook 貼文／影片連結；只保留清除追蹤參數後的 canonical URL。
- Facebook 密碼、cookie、token 與 session 不可寫入 UI、設定檔、log 或 Dataset。
- 保留 Drive file ID、原始檔名、時間碼與 checksum，確保每張影格可追溯。
- 每個 Dataset 必須保存 `sharer_name`、`source_provider`、canonical `source_url`、來源標題與掃描到的影片數。無法從公開連結可靠取得分享者時，由使用者手動填寫。

## 兩種訓練資料

- Detector：獨立 JPEG 與人物 bbox。
- Action：同一 `track_id` 的連續影格或短片段與四項動作標注。

## 切分

Train、validation、test 預設採 80/10/10、seed 42，並按來源影片或時間片段分組後隨機切分。同一片段的相鄰影格不可分散到不同 split。

## 建議 Drive 目錄

```text
SurfTrackDatasets/
  sources/
  datasets/
    surftrack-v0001/
      frames/
      clips/
      annotations/
      splits/
      manifest.json
```

不要把所有影格放在同一個 Drive 資料夾；後續依來源影片與資料版本分層。
