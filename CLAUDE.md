# SurfTrack 開發約定

使用者當次要求、安全規則優先於本文件。專案背景與流程見 [README.md](README.md)，
資料與標注規格見 [docs/spec.md](docs/spec.md)。

## 常用指令

```bash
.venv/bin/pytest -q                 # 全部測試，約 5 秒
.venv/bin/pytest tests/test_store.py -q
./surftrack.py                      # 本機 UI，127.0.0.1:8080
```

## 這個 repo 的慣例

- 只用 stdlib + 原生 JS 寫 server 與前端；torch 只在 `training/` 用，且是 optional
  dependency，import 不可洩漏到 server 路徑上。
- UI 與 API 的錯誤訊息用繁體中文，程式碼、註解、commit message 用英文。
- 所有執行期資料（DB、影片、影格、模型、憑證）只寫進 `var/`，不進 git。
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
