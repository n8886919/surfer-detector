# SurfTrack

SurfTrack 是一個個人開源專案，用來建立可部署至 Jetson Orin Nano 8GB 的 surfer 即時辨識模型。

目前固定範圍：

- 單路相機，模型處理目標 10 Hz。
- 找出畫面中所有 surfer，bbox 只框人物身體。
- 每個 bbox 輸出「追浪／起乘／衝浪」三項獨立機率。
- 不負責選擇跟拍目標、控制相機或辨識人物身分。
- 來源可為公開 Google Drive，或目前登入帳號可查看的 Facebook 貼文。

## 安裝與啟動 UI

只需要 Python 3.11；專案不使用大型 Web framework：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
./surftrack.py
```

開啟 <http://127.0.0.1:8080>。

若系統已經將 `python` 指向 Python 3，也可以使用：

```bash
python ./surftrack.py
```

目前這台開發環境沒有 `python` alias，所以實際使用 `./surftrack.py` 或 `python3 ./surftrack.py`。

安裝時會加入小型 `gdown` 依賴，用來直接列出及下載「知道連結的任何人」可存取的 Drive 來源，不需要來源 Google 帳號或 API key。若尚未安裝依賴，UI 仍可驗證 Drive 連結格式，但不會假裝已經掃描影片。

Facebook 連結輸入目前會完成格式辨識、移除 `__cft__`、`__tn__` 等追蹤參數，並拒絕任何帶有 token、cookie 或 session 的網址。尚未接入 Facebook 登入與影片下載；後續只允許 SurfTrack 專用的隔離瀏覽器 profile，不讀取日常瀏覽器 cookies，也不把 cookies 寫入專案。

## 私人 Google Drive 憑證

- 不把 OAuth token 貼進 UI，也不放在一般 `.env`。
- 預留的本機位置是 `var/secrets/rclone.conf`；`var/` 與 `rclone.conf` 都在 `.gitignore`，目錄權限為 `700`。
- `gitignore` 只防止誤提交，不等於加密。接上 rclone 時仍會啟用 config encryption；解密密碼不可與 config 放在同一檔案。
- 目前尚未建立 token 或 rclone config，也沒有讀取任何 Google 帳號憑證。

## 測試

```bash
pytest -q
```

## 目前狀態

- 已完成工程師極簡的 Data／Label／Train 單頁 UI；掃描後會直接指出下一步。
- 已完成公開 Drive 掃描，以及 Facebook 貼文連結的安全接收與正規化。
- 建立 Dataset 時必須填分享者／來源名稱，並保存來源平台、canonical URL、來源標題及影片數。
- Label 支援拉框、W/E/R 多選狀態、點框重設、Delete 刪除與空白鍵保存下一張。
- SQLite 會保存工作 checkpoint、Dataset、影格與標注；意外關機後執行中的工作會標為可恢復。
- Train 頁面已定義 detector recall、mAP50、action macro F1、三類 F1 與 Orin Hz。
- 尚未接上私人 Google Drive OAuth、Facebook 隔離登入、實際抽圖 worker 與模型訓練 worker。

## 部分標注

你可以只框一張圖裡的部分 surfer。這些框可以訓練三種動作分類；但圖片預設不會訓練 bbox detector，避免未框的 surfer 被誤當背景。只有在 Label 頁勾選「區域內所有 surfer 都已框完」後，該圖片才進入 detector dataset。

標注定義見 [docs/labels.md](docs/labels.md)，資料原則見 [docs/data.md](docs/data.md)。
