# NLP HW3 作業規定與提醒（NOTE）

學號 115062556　｜　Kaggle username 17568160　｜　繳交檔名 `NLP_HW3_NTHU_115062556.ipynb` / `NLP_HW3_NTHU_115062556.zip`

> ## ⚠️ 本文件的每一條都必須嚴格遵守
> 作業說明、助教公告、討論區回覆、助教 email 回覆，**任何一條違反都會被扣分，其中有幾條是大幅扣分、甚至 0 分**。
> 每條規定都標了來源。與助教講好的口頭 / email 回覆，視同規定。**有疑問先寄信問助教，不要自行假設。**
> 助教聯絡信箱（作業說明 PDF 第 30 頁）：`nthuikmlab@gmail.com`

---

## 0. 繳交前必做清單（照順序打勾）

截止前最後一次檢查，**每一項都要確認**。狀態是 2026-11-21 時的狀態。

| 狀態 | 項目 | 說明 |
|---|---|---|
| ☑ | `NLP_HW3_NTHU_115062556.ipynb` 用 `hw3` kernel 從頭執行，輸出已保存在檔案裡 | 2026-11-21 重構為 `sample_code.ipynb` 的 38 格架構（Markdown 不動、補充只寫在 Report），以 `jupyter nbconvert --execute` 重新執行，19 個 code cell 無錯誤 |
| ☑ | 用 `sha256sum` 確認 `output/final/submission_v9_m12_share*.csv` 與 `output/v9/` 送出的檔案相同 | 2026-11-21 重跑後 share 1.25 / 1.5 / 1.75 / 1.875 / 2.0 / 2.25 六份 sha256 與重構前全部相同 |
| ☑ | **用本人帳號送出** `output/final/submission_v9_m12_share1.75.csv`、`share2.0.csv`（最終兩份必須是本帳號的提交） | 2026-11-19 已送出：0.889 / 0.888（另送 1.875：0.888） |
| ☐ | **在 Kaggle 手動選 2 份最終提交**（見 §3.2） | 截止：2026-11-23 12:00 |
| ☐ | （建議）用本帳號送出 `share1.25`、`share2.25`、`output/v9/sub_v9o_m10_share1.75.csv`，讓 Report 引用的掃描分數都在本帳號上；送出後補回 README 的【待提交】 | Report 的 public 0.889 已與本帳號一致 |
| ☐ | Report 內容再逐字檢查一次（特別是 Q5 的聲明是否屬實） | 見 §2 Report 規定 |
| ☐ | （建議）寄信向助教確認 `joeddav/xlm-roberta-large-xnli`、`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` 也屬於「已微調過的 checkpoint、< 1B 可用」 | 11/14 的回覆是針對 Erlangshen；寄信由本人決定 |
| ☐ | Windows 端複本同步定稿 notebook | 見 §6 |
| ☐ | zip 裡**只放** `NLP_HW3_NTHU_115062556.ipynb`（沒有其他檔案、沒有資料夾） | 見 §1 |
| ☐ | 檔名逐字檢查（學校 `NTHU`、學號 `115062556`、副檔名） | 錯一個 −5 |
| ☐ | 上傳 NTU COOL | 作業說明寫「三週」，**實際截止時間以 NTU COOL 頁面為準** |
| ☐ | 11/23 12:00 公布 private 後，補填 Report 的 `Your private leaderboard score:` | 在 COOL 截止前補，見 §2 |
| ☑ | Report 題目與格式未更動（用腳本逐字比對 sample code 的 Report cell） | 5 題逐字保留、`Answer:` 5 個 |
| ☑ | Report 已寫 Kaggle username 與 public 分數 | 17568160 / 0.889 |
| ☑ | GenAI 使用聲明寫在 **Report 的 Q5**（notebook 開頭不放說明，Markdown 保持 sample 原文），並在第一個 code cell 留兩行**程式註解**指向 Q5 | 見 §1 |
| ☑ | 沒有呼叫外部 API；每個模型 < 1B 參數；只用 cfever 資料訓練 | 見 §3 |

---

## 1. 作業說明（PDF `NLP_HW3_Fact_Verification.pdf`）

### 1.1 時程與評分
- Kaggle 競賽：**2026-11-04 09:00 開始 → 2026-11-23 12:00 結束**；**private leaderboard 在 2026-11-23 12:00 公布**。
- 作業說明：**三週**內完成，上傳到 **NTU COOL**。（實際截止時間以 COOL 為準。）
- 評分：
  - Coding work **30%**（TODO 1 檢索頁面 10%、TODO 2 檢索證據 10%、TODO 3 分類 10%）
  - Leaderboard **30%**（**private 分數越高，分數越多；public 與 private 都會納入計算**）
  - Report **40%**：Kaggle username、public & private 分數、Q1（5% + 5%）、Q2（5% + 5%）、Q3（5% + 5%）、Q4（10%）、Generative AI Usage Statement
- 資料：CFEVER，train 24012 / dev 3000 / test 6000 題。三階段：Document Retrieval → Evidence Retrieval → Classification（Supports / Refutes / NEI）。
- Kaggle 的 notebook 每個帳號每週 GPU 上限 30 小時（我們在本機跑，不受影響）。

### 1.2 繳交格式與檔名（Delivery policies）
- Coding work **只能是 Python 檔 `.ipynb`**，**不允許其他格式**。
- 上傳前要**打包成 zip**。
- 檔名規則：

| 內容 | 檔名規則 | 我們的檔名 |
|---|---|---|
| Coding work | `NLP_HW3_school_student_ID.ipynb` | `NLP_HW3_NTHU_115062556.ipynb` |
| Zip 檔 | `NLP_HW3_school_student_ID.zip` | `NLP_HW3_NTHU_115062556.zip` |

### 1.3 扣分規則（Punishments）

| 違反的規定 | 扣分 |
|---|---|
| code 檔名不是 `NLP_HW3_school_student_ID.ipynb`（**只有 .ipynb 可接受**） | **−5** |
| zip 檔名不是 `NLP_HW3_school_student_ID.zip` | **−5** |
| **沒有提供 Kaggle username** | **−10** |
| **沒有在報告中說明 Generative AI 使用** | **−10** |
| code 或 report 與其他同學高度相似 | **兩邊都 −100** |

### 1.4 程式規範（Rules of coding）
- **使用 ChatGPT 或任何 Generative AI，必須同時在「程式註解（Code comments）」與「報告（Report）」兩處說明用途。**
- **禁止抄襲**：不可複製貼上同學的程式；**程式或報告重複 → 0 分**。
- **從網路上參考程式碼，必須提供連結。**
- 以下三種行為會扣該次作業的分數：(1) 使用 Generative AI 而未說明；(2) 使用網路來源而未說明；(3) 抄襲。

### 1.5 Kaggle 提交檔格式
- CSV 三欄：`id`、`predicted_label`、`predicted_evidence`；**保持 test 的 id 與順序**。
- `predicted_label` 是 `supports` / `refutes` / `NOT ENOUGH INFO`；證據是 `json.dumps()` 後的 `[[page_title, sentence_id], …]` 字串（前 5 句）；**NOT ENOUGH INFO 的證據填 `[]`**。
- 分數是 CFEVER（FEVER）score：標籤對 + 證據完整涵蓋任一組標準證據（NEI 只看標籤）。

---

## 2. Report 規定（最嚴格，**會被腳本自動解析**）

### 2.1 助教的作業公告（Report 格式）
> 從作業三開始，如果發現同學們有更動 Report Markdown 格式的話（自己重寫題目 / 整個格式改動），**將會進行扣分**。
> 因為修課人數眾多，為加速批改，維持 report format，**script 才會成功抓出你們的答案**。
> **請直接在每一題的 `Answer:` 後面直接進行作答。**

### 2.2 討論區問答（同學提問 / 助教回答）
- **問**：可否提供 script 讓我們自己先測試能不能把答案抓出來？
  **助教答**：只要你只是在 `Answer:` 冒號後面直接作答，並且**不要動我們給的題目**，就都可以成功被 parse。
- **問**：如果答案是在冒號的**下面**，script 會有問題嗎？
  **助教答**：不會，只要不要動 format，直接在此題的 `Answer:` 和**下一題的問題中間**作答，都可以。

### 2.3 我們必須做到的事
1. **不要改動任何題目文字與格式**：`# Report`、`Your Kaggle username:`、`Your public leaderboard score:`、`Your private leaderboard score:`、Q1～Q5 的題目、`Answer:` 全部維持 sample code 的原文（連 Q2、Q3 題目裡的雙空白也一樣）。
2. **所有作答內容只能放在某題的 `Answer:` 與下一題題目之間。**
3. 不要在作答內容裡寫出像「`2. xxx`」這種會被誤認成題目開頭的行；不要在答案裡再出現 `Answer:`；不要加額外的 `# ` 標題。（目前的 Report 已用腳本檢查過。）
4. **不要加額外說明行**（例如 v1 版曾多加「（以下內容待實驗完成後填寫）」，已移除）。
5. Report 必須包含：**Kaggle username**（缺少 −10）、**public 與 private 分數**、Q1～Q4、**Generative AI Usage Statement**（缺少 −10）。
6. Report 位在 notebook 的**最後一個 cell**（與 sample code 相同位置），是 Markdown cell。
7. **private 分數**：11/23 12:00 才公布，我們目前留白；公布後補填。若在 COOL 截止前無法補，保持原格式留白，不要改動欄位。

### 2.4 Report 各題的要求（取自題目）
- Q1：描述**檢索 Wikipedia 文件的方法**（5%），並回報 **development set 上的 Recall@N**（5%）。
- Q2：描述**檢索證據的方法**（5%），並回報 **development set 上的 Recall@N**（5%）。
- Q3：描述**分類方法**（5%），並回報 **development set 上的 Accuracy**（5%）。
- Q4：「… Anything that can strengthen your report.」（10%）——消融實驗、負面結果、分析。
- Q5：Generative AI Usage Statement。

---

## 3. 助教公告與模型 / 資料規定

### 3.1 助教公告（違反第 1、2 條**大幅扣分**）
1. **只能使用「每個模型」參數量 < 1B 的預訓練模型**（集成時，每個成員各自 < 1B，不累計）。
2. **禁止呼叫外部 API**（指推論服務；下載模型權重 / 資料集、用 Kaggle CLI 提交都可以）。
3. 三個 stage 的訓練架構**可以自訂**。
4. 違反第 1、2 條**大幅扣分**。

### 3.2 Kaggle 最終提交（**截止前必須手動選**）
- **最多選 2 份** submission 作為最終計分版本。
- **最終兩份（2026-11-17 決定，取代 11-14 的 v7 兩份）**：
  1. `output/final/submission_v9_m12_share1.75.csv` — 12 成員集成，NEI 比例對齊 v7 偏差 1.75（A 組 52.3%）
  2. `output/final/submission_v9_m12_share2.0.csv` — 同上，對齊 v7 偏差 2.0（A 組 55.6%）
  - 理由：12 成員集成的 NEI 比例在 1.75～2.0 形成平台、2.25 開始下降；兩份取平台兩端，private 取兩者較高者。
- 這兩份**必須先用本人帳號送出**，才能在 "Select final submissions" 頁面被勾選；勾選前先看檔名與分數欄。
- v7 的兩份（0.881 / 0.876）與 v8 的 refutes 偏差檔（0.875）**不再選**。

### 3.3 討論區（2026-11-11，「Starter Code 的修改範圍」）
- sample code **只是範例**，**只要合乎 HW3 的規定（bullets）即可任意修改**。
- **「也只能使用 cfever dataset」**——訓練只能用 CFEVER 提供的 train / dev / wiki。
- **「不要更動 Report Markdown 格式」**（見 §2）。

### 3.4 助教 email 回覆（2026-11-14）
- **我們問的**：Erlangshen 系列（`IDEA-CCNL/Erlangshen-Roberta-110M-NLI`、`…-330M-NLI`）這類**已在外部 NLI 資料上預訓練過的 checkpoint**，能不能用？
- **助教答**：**「可以的，只要參數量 1b 以下即可」**。
- 信是**使用者自己寄出**的；**沒有使用者允許，不可替他寄信或發送任何 email。**
- 因此：**不可使用超過 1B 的模型**，例如 `Erlangshen-MegatronBert-1.3B-NLI`（1.3B）——**嚴禁使用**。

### 3.5 我們使用的模型（全部 < 1B，Report 的 GenAI 段已逐一列出）

| 模型 | 參數量 | 用途 |
|---|---|---|
| `BAAI/bge-small-zh-v1.5` | 約 24M | 稠密頁面檢索 |
| `hfl/chinese-macbert-base` | 約 102M | 句子重排器、分類器成員 |
| `hfl/chinese-roberta-wwm-ext` | 約 102M | 分類器成員（v6 另用作第二個重排器） |
| `IDEA-CCNL/Erlangshen-Roberta-110M-NLI` | 約 110M | 分類器成員 |
| `IDEA-CCNL/Erlangshen-Roberta-330M-NLI` | 約 326M | 分類器成員（本機兩個種子；Kaggle 上另訓練 2 個附兄弟 claim 的版本） |
| `Qwen/Qwen3-0.6B-Base`（+ LoRA） | 約 0.6B | 分類器成員（完整版 + 附兄弟 claim 版；v7 的 pilot 已被取代） |
| `joeddav/xlm-roberta-large-xnli` | 約 560M | 分類器成員（v9，Kaggle GPU；2 個版本） |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 約 278M | 分類器成員（v9，Kaggle GPU） |

- 未呼叫任何外部 API；訓練資料只用 cfever 的 train / dev / wiki（模型本身的預訓練資料不計，已向助教確認）。
- Kaggle Notebook GPU 只當作運算資源（私人 notebook + 私人 dataset `yiting224/cfever-hw3-evidence`，內容是由本 notebook 匯出的 cfever 資料與檢索結果），不是推論 API。

---

## 4. Generative AI 與引用（兩處都要寫，缺一不可）

| 位置 | 要求 | 目前狀態 |
|---|---|---|
| **Report 的 Q5** | 說明使用了哪些 Generative AI、做了什麼、使用的預訓練模型、參考資料 | ✅ 已寫（Claude Sonnet 5.5（v0–v8）與 Opus 5.5（v9）：架構設計、程式碼、Kaggle GPU 訓練程式、統計分析、除錯、Report；列出所有模型（含 XLM-R / mDeBERTa XNLI）與助教回覆） |
| **程式註解（Code comments）** | 在程式裡也要說明 | ✅ 第 1 個 code cell 開頭兩行註解（「Generative AI 使用聲明…」，指向 Report Q5）；notebook 的 Markdown 不放聲明 |
| **網路來源的連結** | 參考的程式碼 / 資料要給連結 | ✅ CFEVER 資料集與論文（`https://github.com/IKMLab/CFEVER-data`）、Hugging Face 各模型 model card、作業 sample code |

> ⚠️ Report Q5 寫了「所有程式由本人實際執行」「Report 內容由本人對照輸出檢查過」。**這些必須是事實**——繳交前請你自己確實檢查過 Report 與 notebook。

---

## 5. 誠信與其他「不要做」

- **不要與同學共用程式或報告**，也不要抄同學的（重複 → 兩邊都 −100，或 0 分）。
- **不要違反模型規定**：每個模型 < 1B、不呼叫外部 API、只用 cfever 資料訓練。
- **不要在 Report 動格式**、不要多加說明行、不要重寫題目。
- **不要漏選 Kaggle 最終 2 份提交**。
- **不要寄信給助教前沒讓使用者確認**（寄信由使用者決定與發送）。
- **不要擅自把 `.ipynb` 以外的東西放進 zip**（資料、`output/`、CSV、metrics.json 都不要放）。
- **不要用超過 1B 的模型**（例如 `Erlangshen-MegatronBert-1.3B-NLI`）。
- 不確定的規定先寄信問助教 `nthuikmlab@gmail.com`，**拿到書面回覆再行動**。

---

## 6. 這個專案的檔案與環境（備忘）

- 專案：WSL2 Ubuntu 的 `~/NLP-Assignment3`（venv `~/venvs/hw3`，RTX 3050 4GB）；Windows 端複本 `C:\Users\user\Desktop\NLP\NLP-Assignment3`，**兩份不會自動同步**，記得把定稿 notebook、輸出檔複製回 Windows。
- VS Code 重開機後：notebook 右上角 Select Kernel → Python Environments… → `hw3 (~/venvs/hw3/bin/python)`；新 kernel 要先 Run All 才有變數。
- 輸出位置：`output/<版本>/`（v6 起）；快取 `output/cache/`（共用，不要刪，否則所有模型要重算）。
- 實驗紀錄與各 csv 的 public 分數：見 `README.md`。Report 草稿：專案 `claude/hw3_report_draft.md`。
