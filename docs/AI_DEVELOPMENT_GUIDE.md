# AI Development Guide

本文件適用於在 SurfTrack 中工作的 Codex 與 subagent。
使用者當次要求、系統安全規則及明確的專案限制優先於本文件。

## 1. 工程等級

SurfTrack 是個人 side project，不是大型企業產品。

目標：

> 用最少合理工程成本，做出正確、易懂、容易維護的實作。

除非任務確實需要或使用者明確要求，不要主動導入：

- enterprise architecture
- ADR、RFC 或大型 design document
- 複雜 design pattern 或過度 abstraction
- repository-wide refactor
- 完整 CI/CD、release governance 或多層審批流程
- 完整 migration framework
- 大量 boilerplate、文件或註解
- 全面 unit、integration、E2E test
- 為假想未來需求預先設計 extensibility

## 2. 修改優先順序

固定使用以下升級順序：

`最小修正 > 小範圍改善 > 局部重構 > 架構重設`

只有前一層無法合理解決問題時，才進入下一層。

優先選擇：

- existing code > new abstraction
- existing dependency > new dependency
- simple code > clever code
- fewer files > more files
- easier debugging > theoretical elegance
- current requirement > hypothetical future requirement

## 3. Token 與 Context

- 只讀取完成任務需要的檔案。
- 小任務不要掃描整個 repository。
- 不重複讀取已確認的資訊。
- 找到足夠證據後停止搜尋。
- 不探索與目前問題無關的程式。
- 不重述使用者需求。
- 不輸出冗長 planning；小任務直接執行。
- 避免 plan、implementation、review、summary 重複描述同一內容。
- 不為了看起來完整而新增文件、抽象層或流程。

## 4. Subagent

Subagent 不是預設步驟。簡單修改由主 agent 直接完成。

只在工作能明確切割，且可降低主 agent context 污染時使用，例如：

- 搜尋特定 implementation
- isolated code review
- bug 或 edge-case hunting
- test failure analysis
- log analysis
- narrow research task

若可選模型，採成本導向策略：

- 一般搜尋或簡單分析：Luna Medium
- 困難但範圍明確的 review 或 debugging：Luna Max
- 只有真正需要大型架構推理時才使用更昂貴模型

不要：

- 讓多個 agent 重複分析同一問題
- 無理由 parallel spawn agents
- 對每次修改執行多層 agent review

## 5. 測試

測試目標：

> 合理確認這次修改正確，且沒有破壞直接相關功能。

- 優先執行與修改直接相關的既有測試。
- 小修改不需要順便建立完整 test framework。
- 修 bug 時，若 regression test 容易建立且有價值，可以增加。
- 不為單一小 feature 建立大量 mock、fixture 或 infrastructure。
- 不為 coverage 數字撰寫低價值測試。
- 測試深度應與修改風險相稱。

## 6. 文件與註解

只維護真正有用的文件。

除非專案原本要求或使用者明確指定，不要因每次修改自動：

- 更新 architecture document
- 建立 ADR 或 changelog entry
- 建立 implementation report
- 建立 handoff document
- 產生其他大量 Markdown 文件

Code 能清楚表達意圖時，不寫冗長 comment。

## 7. Scope Discipline

發現與任務無關的問題時：

- 不順手重構
- 不順手 cleanup
- 不順手改 formatting
- 不擴大 scope

若發現嚴重 bug、安全風險或資料損失風險，可以簡短指出，
但不要在未獲授權時擴大修改範圍。

保留使用者既有修改；只變更完成任務必要的檔案與行數。

## 8. 何時提高工程深度

以下問題不可為節省 token 而草率處理：

- security
- authentication 或 authorization
- data loss
- destructive operation
- database 或 schema migration
- concurrency 或 race condition
- protocol 或 API compatibility
- irreversible changes
- 核心 architecture decision
- 原因不明且可能造成大範圍影響的修改

遇到上述情況，才提高 reasoning、testing 或 review 深度，
並只增加足以控制實際風險的工程量。

## 9. 完成回報

一般任務完成後只回報：

1. Changed
2. Validation
3. Risks / unresolved issues（只有存在才列）

不要附上完整工作日誌，也不要重複已知背景。
