# SurfTrack — Codex 專案指引

使用者當次要求、安全規則優先於本文件。專案背景與流程見 [README.md](README.md)，
資料與標注規格見 [docs/spec.md](docs/spec.md)。

## 開始工作

- 使用繁體中文溝通。先用 `git status --short` 與任務相關的 diff 確認工作區狀態，
  保留既有的未提交修改；查找檔案與文字優先用 `rg --files`、`rg`。
- 只讀任務需要的文件與程式碼，以目前 Git、實際檔案與驗證結果為準。
  記憶或前次對話可作線索，但資料筆數、訓練狀態與遠端環境必須重新確認。
- 已授權且可逆的工作直接完成並驗證；使用者說「先討論」時只分析，不修改。

## 開發環境與常用指令

Python 3.11 以上，優先使用專案 `.venv`，不要改動系統 Python。
全新環境才需要建立 virtualenv 與安裝依賴：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . pytest
```

抽影格需要 PATH 中有 `ffmpeg`。只有任務需要本機訓練依賴時才安裝
`.[train]`；遠端 CUDA 主機設定見 README，不假設本機有 GPU。

```bash
.venv/bin/pytest -q                 # 全部測試
.venv/bin/pytest tests/test_store.py -q
./surftrack.py                      # 本機 UI，127.0.0.1:8080
```

## 這個 repo 的慣例

- 只用 stdlib + 原生 JS 寫 server 與前端；torch 只在 `training/` 用，且是 optional
  dependency，import 不可洩漏到 server 路徑上。
- UI 與 API 的錯誤訊息用繁體中文，程式碼、註解、commit message 用英文。
- 所有執行期資料（DB、影片、影格、模型、憑證）只寫進 `var/`，不進 git。
- 不輸出或提交 token、OAuth 資料、rclone 設定內容；測試使用暫存資料，避免改動真實標注。
- Schema 變更走 `store.py` 裡的 `_migrate_*` 函式，就地 ALTER，不引入 migration
  framework；`var/surftrack.sqlite3` 是真實資料，改動前先備份到 `var/backups/`。
- 長時間工作（ingest、train）跑在背景 thread 並把進度寫回 SQLite，重啟後要能標為可恢復。

## 工程等級

個人 side project，不是企業產品。目標是**用最少合理成本，做出正確、易懂、好維護的實作**。

除非任務真的需要或使用者要求，不要主動導入 ADR／design doc、複雜抽象、repo-wide
refactor、CI/CD governance、大量 boilerplate 或為假想未來需求預留的擴充點。

修改用固定升級順序：`最小修正 > 小範圍改善 > 局部重構 > 架構重設`，前一層無法合理
解決才進下一層。偏好 existing code > new abstraction、existing dependency > new
dependency、fewer files > more files、current requirement > hypothetical future。

## Scope

發現與任務無關的問題時不順手重構、cleanup 或改 formatting；嚴重 bug 或安全風險簡短
指出即可，未獲授權不擴大修改。保留使用者既有的未提交修改。

## 測試與文件

測試目標是合理確認這次修改正確、沒破壞直接相關功能：優先跑相關的既有測試，修 bug 時若
regression test 容易寫就加上，不為 coverage 寫低價值測試，也不為單一小 feature 建大量
mock 與 fixture。

不要每次修改就自動更新架構文件、產生 changelog、implementation report 或 handoff
document。Code 能表達意圖時不寫冗長註解。

## 何時提高深度

以下不可為省 token 草率處理：security、authn/authz、資料遺失、破壞性操作、schema
migration、concurrency／race condition、API 相容性、不可逆變更、核心架構決策。
遇到這些才提高 reasoning、testing 與 review 深度，且只加到足以控制實際風險。

## 回報

完成後回報 Changed / Validation / Risks（只有存在才列），不附完整工作日誌，不重述已知背景。

## 跨 Session 工作

延續工作時先核對使用者目標、Git diff 與相關檔案；缺少必要上下文時只問關鍵問題。
需要交接時，在回覆中簡短列出已完成與已驗證事項、剩餘工作及具體檔案。
只有使用者明確要求時才更新持久記憶或建立交接文件。
