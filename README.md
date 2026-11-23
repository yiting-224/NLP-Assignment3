# NLP HW3 — CFEVER 中文事實查核：實驗紀錄

學號 115062556　｜　Kaggle username 17568160　｜　競賽：`nthu-nlp-2026-assignment-3-fact-verification`（2026-11-04 09:00 ～ 2026-11-26 00:00；助教 11/23 公告延長，並公布 private 快照）

這份文件記錄**所有實驗（成功與失敗）**、每個實驗對應的 notebook 檔名、每個 notebook 輸出了哪些 `.csv`，以及各 `.csv` 在 Kaggle 的 public score。
**從未上傳 Kaggle 的 `.csv`，public score 一律填【未測】。**

- 作業規定與提醒請看 `NOTE.md`。
- 分數＝Kaggle FEVER score（與 notebook 內的 CFEVER 相同）。dev 指標都在 dev 3000 題（supports / refutes / NEI 各 1000）上計算。
- 完整的 Report 草稿存在專案 `claude/hw3_report_draft.md`。

---

## 0. 一眼看完

**目前最終提交（Kaggle 上要勾選的兩份，截止前手動選）：**

| 順序 | 檔案 | 產生它的 notebook | public | NEI 比例 |
|---|---|---|---|---|
| 1 | `output/final/submission_v9_m12_share1.75.csv` | `NLP_HW3_NTHU_115062556.ipynb`（最後一個 code cell：Submission） | **0.889** | 對齊 v7 偏差 1.75（A 組 52.3%） |
| 2 | `output/final/submission_v10_m12_nei0.0.csv` | `NLP_HW3_NTHU_115062556.ipynb`（最後一個 code cell：Submission） | 0.862 | 不加偏差，只用 dev 決定（A 組 34.2%）；見 v10 |

（2026-11-23 起取代 `share2.0.csv`。11/23 的 private 快照：share1.75 / share2.0 那一組是 **0.8346**；原因與修正見 §2 的 v10。）

定稿 notebook：`NLP_HW3_NTHU_115062556.ipynb`（**2026-11-21 重構版**：沿用 `sample_code.ipynb` 的 38 格架構、Markdown 原文不動，19 個 code cell 放 v9 的最終 pipeline；補充說明只寫在最後的 Report；**已用 hw3 kernel 從頭執行並保存輸出**，輸出到 `output/final/`；六份 `submission_v9_m12_share*.csv` 重跑後 sha256 與重構前完全相同）。
重構前的版本（自訂架構：v8.0 + 第 9 節 v9 + Report，2026-11-17 執行）保留在 git commit `047ce6e`。

**最終 pipeline（v9）：**
1. 頁面檢索：標題子字串 + BM25（字元 bigram）+ 稠密檢索（`bge-small-zh-v1.5`）→ HistGradientBoosting 學習式排序（完整 wiki 119 萬頁），前 6 頁。（與 v7 相同）
2. 證據檢索：bigram 粗篩 32 句 → MacBERT cross-encoder 重排 → 證據句選擇器（HGB，dev 5 折 OOF）→ 前 5 句。（與 v7 相同）
3. 分類：**12 成員集成**（v7 的 5 個 + 完整版 Qwen3-0.6B LoRA + 附兄弟 claim 的 Qwen3-0.6B + 5 個在 Kaggle GPU 上訓練的成員：XLM-R-large-XNLI、Erlangshen-330M-NLI × 2（附兄弟 claim）、XLM-R-XNLI（附兄弟 claim）、mDeBERTa-v3-XNLI（附兄弟 claim）），claim + 前 4 句，log-softmax 等權平均 + NEI 偏差（最終之一照 public 對齊 NEI 比例；最終之二不加偏差，見 v10）。

**dev 指標演進：**

| 版本 | notebook | 頁面 Recall@5 | 證據 Recall@5 | 分類 dev accuracy | dev CFEVER（未加偏差） |
|---|---|---|---|---|---|
| v1 | `…v1.ipynb` | 0.9175 | 0.8615 | 0.7137（單一 macbert） | （未保存） |
| v2 | `…v2.ipynb` | 0.9175 | 0.8615 | 0.7353 | （未保存） |
| v3.0 | `…v3.0.ipynb` | 0.9175 | 0.8615 | 0.7573 | 0.7130 |
| v3.1 | `…v3.1.ipynb` | 0.9175 | 0.8615 | 0.7723 | 0.7280 |
| v4.0 | `…v4.0.ipynb` | 0.9175 | 0.8615 | 0.7720 | 0.7273 |
| v5.0 | `…v5.0.ipynb` | 0.9175 | 0.8780 | 0.7720 | 0.7330 |
| v6.0 | `…v6.0.ipynb` | 0.9175 | 0.8780 | 0.7720 | 0.7330 |
| v7.0 | `…v7.0.ipynb` | 0.9175 | 0.8780 | 0.7780 | 0.7370 |
| v8.0 | `…v8.0.ipynb` | 0.9175 | 0.8780 | 0.7780 | 0.7370（主流程未變） |
| v9 | `NLP_HW3_NTHU_115062556.ipynb` | 0.9175 | 0.8780 | **0.7963** | **0.7550** |

（`…` ＝ `NLP_HW3_NTHU_115062556`。v0 因分類器捷徑 bug，dev 數字沒有保存；v1、v2 只保存了 accuracy（取自成員表 / 子集合表），當時的 CFEVER 沒有保存。v3.0 起的數字取自各版的 `metrics.json`。）

---

## 1. 輸出檔案與位置的約定

- 提交檔檔名格式：`submission.csv`（主輸出）、`submission_nocal.csv`（未校準）、`submission_{成員代號}_cal.csv`（含 dev 校準）、`submission_{成員代號}_nei{偏差}.csv`（全域 NEI 偏差）、`submission_{成員代號}_Anei{偏差}.csv`（只對 test A 組加偏差）。
- 成員代號：`m5_d13d8f` ＝ 5 成員（含 qwen pilot）、`m6_037f63` ＝ 6 成員；`_sel` 表示有用證據句選擇器。
- v0～v5.1：輸出在 `output/`（扁平）；v6 起：`output/<版本>/`；快取固定在 `output/cache/`。
- 「本機保存檔名」是我手動重新命名後存放的檔名（例如 `submissionv0.csv`）；v3 以前程式輸出的檔名都叫 `submission.csv`，會互相覆蓋，所以是靠我當時改名保存。
- **「【未測】」＝這個檔案從未上傳過 Kaggle**（不論有沒有保存在本機）。每個 notebook 也都會輸出 `metrics.json`（不是 csv，不列入表中）。
- 各檔的版本歸屬已用 CSV 內的 NEI 筆數核對過（例如 `nei2.0v3.1` 的 NEI 2689 筆、`nei3.0v3.1` 3196 筆，確認是 v3.1 而不是 v3.0）。

---

## 2. 逐版實驗紀錄

### v0 — 第一個完整 pipeline（失敗：分類器捷徑 bug）

- **notebook**：`NLP_HW3_NTHU_115062556v0.ipynb`
- **內容**：改用完整 wiki（119 萬頁）而非 sample code 的「只索引訓練頁面」（dev 的證據頁面只有 10.8% 在其中）；頁面檢索（標題 + BM25 + 稠密 → HGB 排序）、句子 cross-encoder（MacBERT）、超連結第二跳；MacBERT 分類器讀 claim + 前 5 句。
- **結果**：public 0.716，**失敗**。
- **原因**：NOT ENOUGH INFO 的訓練證據永遠是 4～5 句，supports / refutes 則是「標準證據 + 隨機 0～3 句干擾」，約一半不足句數，且標準證據永遠排第一。模型學成「滿句數 ＝ NEI」；測試時每題都是滿句數，dev 上 43% 的 supports / refutes 被判成 NEI。

| 程式輸出的 csv（`output/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission.csv` | `submissionv0.csv` | v0 預測 | 0.716 |
| `submission_quick.csv` | — | 只有 `QUICK_TEST=True` 時才輸出（快速檢查流程用） | 【未測】 |

### v1 — 修正捷徑（成功）

- **notebook**：`NLP_HW3_NTHU_115062556v1.ipynb`
- **改動**：三類標籤的分類訓練證據都改成「同一個重排器對訓練題實際重排的前 4 句」（30% 機率打亂順序；supports / refutes 若標準證據不在前 4 句，就補進去並擠掉分數最低的非標準句）。分類器輸入改為 claim + 前 4 句。加入 dev 校準（對各類別 logit 加偏差）。
- **結果**：public 0.716 → **0.833**。dev：單一 macbert accuracy 0.7137。

| 程式輸出的 csv（`output/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission.csv` | `submissionv1.csv` | 校準版（NEI +0.75） | 0.833 |
| `submission_nocal.csv` | `submission_nocalv1.csv` | 未校準 | 0.809 |

### v2 — 2 成員集成（成功，幅度小）

- **notebook**：`NLP_HW3_NTHU_115062556v2.ipynb`
- **改動**：分類器改成集成（`macbert_base` + `roberta_base`），log-softmax 後等權平均。
- **結果**：dev accuracy 0.7137 → 0.7353；public 0.810，與 v1 未校準版（0.809）相比只多約 0.1 pt（v1 的 0.833 是校準版，不能直接比較）。

| 程式輸出的 csv（`output/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission.csv` 與 `submission_nocal.csv` 兩者之一 | `submissionv2.csv` | 依專案筆記為 2 成員未校準版 | 0.810 |
| 另一個（校準版或未校準版，未保存） | — | — | 【未測】 |

### v3.0 — 加入 NLI 預訓練成員（成功）＋ NEI 偏差掃描

- **notebook**：`NLP_HW3_NTHU_115062556v3.0.ipynb`
- **改動**：加入 `erlangshen_110m`（`IDEA-CCNL/Erlangshen-Roberta-110M-NLI`，輸入把證據放前面、分類頭依 蘊含/矛盾/中立 → supports/refutes/NEI 重排）。新增「NEI 偏差掃描」：對 NOT ENOUGH INFO 的 logit 加偏差並各存一份。
- **結果**：成員 acc：110m 0.7470（同為 RoBERTa-base 規模，比 `roberta_base` 0.7227 多 2.4 pt）；3 成員集成 dev accuracy 0.7573、CFEVER 0.7130；dev 校準未通過交叉驗證，不採用（偏差 [0,0,0]）。public 隨 NEI 偏差上升：0 → 0.824、0.5 → 0.844、1.0 → 0.852、1.5 → 0.861。

| 程式輸出的 csv（`output/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission.csv` | `submissionv3.csv` | NEI 偏差 0 | 0.824 |
| `submission_nocal.csv` | — | 與 `submission.csv` 相同（未採用校準） | 【未測】 |
| `submission_nei0.5.csv` | `submission_nei0.5v3.csv` | NEI +0.5 | 0.844 |
| `submission_nei1.0.csv` | `submission_nei1.0v3.csv` | NEI +1.0 | 0.852 |
| `submission_nei1.5.csv` | `submission_nei1.5v3.csv` | NEI +1.5 | 0.861 |
| `submission_nei2.0.csv` | — | NEI +2.0（v3.0 版） | 【未測】 |
| `submission_nei3.0.csv` | — | NEI +3.0（v3.0 版） | 【未測】 |

### v3.1 — 加入 330M NLI 成員（成功）

- **notebook**：`NLP_HW3_NTHU_115062556v3.1.ipynb`
- **改動**：加入 `erlangshen_330m`（`IDEA-CCNL/Erlangshen-Roberta-330M-NLI`，約 326M）。4GB 顯存放不下完整 AdamW，改用 Adafactor、凍結 embedding 與前 8 層（OOM 時自動多凍 4 層重試）。
- **結果**：330M 單模型 acc 0.7663（比 110M 多 1.9 pt）；4 成員 dev accuracy 0.7723、CFEVER 0.7280；dev 校準（偏差 [0, 0.75, 1.0]）通過交叉驗證 → 採用，但 **public 只有 0.851**，反而低於同模型的 NEI +2.0（0.873）。
- **教訓**：dev 上校準出來的 NEI 偏差（+1.0）明顯低估 public 偏好的偏差（約 +1.5～2.0）——dev 與 test 的類別比例不同（見「教訓」）。
- 這一版的 NEI 偏差曲線：2.0 → 0.873、3.0 → 0.854，峰值在 2.0 附近。

| 程式輸出的 csv（`output/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission.csv` | `submissionv3.1.csv` | dev 校準偏差 [0, 0.75, 1.0] | 0.851 |
| `submission_nocal.csv` | — | 未校準 | 【未測】 |
| `submission_nei0.5.csv` | — | NEI +0.5（v3.1 版） | 【未測】 |
| `submission_nei1.0.csv` | — | NEI +1.0（v3.1 版） | 【未測】 |
| `submission_nei1.5.csv` | — | NEI +1.5（v3.1 版） | 【未測】 |
| `submission_nei2.0.csv` | `submission_nei2.0v3.1.csv` | NEI +2.0 | 0.873 |
| `submission_nei3.0.csv` | `submission_nei3.0v3.1.csv` | NEI +3.0 | 0.854 |

> 注意：`submission_nei2.0v3.1.csv`、`submission_nei3.0v3.1.csv` 曾被我誤記成 v3 的檔案，已用 NEI 筆數核對更正為 v3.1。

### v4.0 — Qwen3-0.6B + LoRA pilot（效果小，保留在集成中）

- **notebook**：`NLP_HW3_NTHU_115062556v4.0.ipynb`
- **改動**：加入 `qwen06_pilot`（`Qwen/Qwen3-0.6B-Base` + LoRA r=16、只用 6000 筆訓練資料、1 epoch）。提交檔檔名改含成員代號（`m5_d13d8f`）。
- **結果**：qwen 單模型 acc 0.7103（最弱），5 成員集成 dev accuracy 0.7720、CFEVER 0.7273（跟 4 成員差不多，集成只多約 0.17 pt）；public（NEI +1.5）0.875，高於 v3 的 3 成員 0.861。
- **判斷**：邊際效果小，但不傷害，保留。完整版 `qwen06`（全部訓練資料、2 epoch）寫在設定裡，**沒有執行**。

| 程式輸出的 csv（`output/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission.csv` | `submissionv4.0.csv` | NEI 偏差 0（未採用校準） | 【未測】 |
| `submission_nocal.csv` | — | 未校準 | 【未測】 |
| `submission_m5_d13d8f_cal.csv` | — | dev 校準 | 【未測】 |
| `submission_m5_d13d8f_nei0.0.csv` | — | NEI +0 | 【未測】 |
| `submission_m5_d13d8f_nei0.5.csv` | — | NEI +0.5 | 【未測】 |
| `submission_m5_d13d8f_nei1.0.csv` | — | NEI +1.0 | 【未測】 |
| `submission_m5_d13d8f_nei1.5.csv` | `submission_m5_d13d8f_nei1.5.csv` | NEI +1.5 | 0.875 |

（`submissionv4.0.csv` 在本機有保存，但專案筆記沒有它的分數紀錄；若你確實送過，請補上分數。）

### v5.0 — 證據句選擇器（成功）

- **notebook**：`NLP_HW3_NTHU_115062556v5.0.ipynb`
- **動機**：分類 accuracy 0.77，但 CFEVER 只有 0.73；差距來自前 5 句沒有涵蓋完整的標準證據組（候選池涵蓋約 96% 的題目，Recall@5 只有 86%）。
- **改動**：在 dev 的 supports / refutes 題上，用 HistGradientBoosting 訓練一個句子選擇器重排前 12 句（特徵：重排分數與名次、與第 1 名的分差、頁面機率與名次、同頁在前 5 的句數、與 claim 的 bigram 重疊…），dev 用 5 折（以題目為單位）out-of-fold 評估。**只改變送出的 5 句證據，分類器輸入不變。**
- **結果**：證據 Recall@5 0.8615 → 0.8780（Recall@1 0.4955 → 0.4970、@3 0.8005 → 0.8275）；依標準證據句數：1 句 0.9356 → 0.9442、2 句 0.7856 → 0.8129、3 句 0.6509 → 0.6864。dev CFEVER 0.7273 → 0.7330。public（NEI +1.5）0.875 → 0.878。

| 程式輸出的 csv（`output/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission.csv` | `submission.csv` | NEI 偏差 0 | 0.838 |
| `submission_nocal.csv` | — | 未校準（與上相同） | 【未測】 |
| `submission_m5_d13d8f_sel_cal.csv` | — | dev 校準 | 【未測】 |
| `submission_m5_d13d8f_sel_nei0.0.csv` | — | NEI +0 | 【未測】 |
| `submission_m5_d13d8f_sel_nei0.5.csv` | — | NEI +0.5 | 【未測】 |
| `submission_m5_d13d8f_sel_nei1.0.csv` | — | NEI +1.0 | 【未測】 |
| `submission_m5_d13d8f_sel_nei1.5.csv` | `submission_m5_d13d8f_sel_nei1.5.csv` | NEI +1.5 | 0.878 |

### v5.1 — 全域 NEI +2.0 與「只對 A 組加偏差」（v5.0 多加一個 cell）

- **notebook**：`NLP_HW3_NTHU_115062556v5.1.ipynb`（＝ v5.0 加一個最後的 cell，只需要跑那一個 cell）
- **改動**：test 可以依 id 與題目樣態分成三組（A：id ≤ 31996，3000 題，類似 dev；B：id > 31996 且含「最喜歡」，1000 題，模型 100% 判 NEI；C：其餘 2000 題長句，像抄自維基，預測 ~50/49/1 為 S/R/NEI）。假設只對 A 組加偏差比全域偏差好。
- **結果**：全域 NEI +2.0 → 0.874；只對 A 組 +2.5 → 0.867，**沒有勝過全域偏差**，而且需要用 id 與「最喜歡」做分組，不夠通用（失敗）。

| 程式輸出的 csv（`output/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission_m5_d13d8f_sel_nei2.0.csv` | 同名 | 全域 NEI +2.0 | 0.874 |
| `submission_m5_d13d8f_sel_nei2.5.csv` | — | 全域 NEI +2.5 | 【未測】 |
| `submission_m5_d13d8f_sel_Anei2.0.csv` | — | A 組 +2.0 | 【未測】 |
| `submission_m5_d13d8f_sel_Anei2.5.csv` | 同名 | A 組 +2.5 | 0.867 |
| `submission_m5_d13d8f_sel_Anei3.0.csv` | — | A 組 +3.0 | 【未測】 |

### v6.0 — 第二個重排器 + hard negative mining（失敗）

- **notebook**：`NLP_HW3_NTHU_115062556v6.0.ipynb`；輸出資料夾 `output/v6/`
- **改動**：訓練第二個句子重排器（`chinese-roberta-wwm-ext`、2 epoch、max length 160），負例直接取自第一個重排器對訓練題的高分非標準句（hard negative mining）；選擇器新增 base / base+link / base+link+rr2 三組特徵並以 OOF Recall@5 選擇。
- **結果**：三組特徵的 OOF Recall@5 全部 0.8780，**沒有增益**，不採用（選用最簡單的 base）。因此 v6 的輸出與 v5 相同：public（NEI +1.5）0.878。
- **其他嘗試**：selector 特徵加入「超連結關聯（是否被第 1/2 名句子的超連結指到）」，Recall@5 一樣是 0.8780。

| 程式輸出的 csv（`output/v6/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission_m5_d13d8f_sel_nei1.5.csv` | `output/v6/` 同名 | NEI +1.5 | 0.878 |
| `submission.csv`、`submission_nocal.csv`、`_sel_cal.csv`、`_sel_nei0.0/0.5/1.0/2.0/2.5.csv`、`_sel_Anei2.0/2.5/3.0.csv` | — | 同 v5 系列的其他組合（v6 重跑產生） | 【未測】 |

### v7.0 — 第二顆 330M（seed 42）→ 6 成員（成功；最終提交來源）

- **notebook**：`NLP_HW3_NTHU_115062556v7.0.ipynb`；輸出資料夾 `output/v7/`
- **動機**：想提高 **private** 分數，而不是靠 public 挑檔。集成帶來的增益是真的（可以在 dev 上用配對檢定驗證），而照 public 挑偏差則不是。
- **改動**：加入 `erlangshen_330m_b`（同一個 330M 架構、seed 42；seed 同時改變分類頭初始化、dropout、資料順序與訓練資料的建立順序，兩顆的錯誤不完全重疊）。6 成員 log-softmax 等權平均。
- **結果**：成員 acc 0.7647；6 成員 dev accuracy 0.7780、CFEVER 0.7370（5 成員 0.7330 → 6 成員 0.7370，同一批 dev 題目配對比較變好 34 題 / 變壞 22 題，約 1.6 個標準誤）。public（NEI +1.5）0.878 → **0.881**；（NEI +2.0）0.874 → 0.876。
- **NEI 偏差曲線（全域，public）**：0.5 → 0.856、1.0 → 0.871、**1.5 → 0.881**、2.0 → 0.876、2.5 → 0.870；二次擬合頂點約 1.7，1.5～2.0 在高原內。dev 上同一組偏差的 CFEVER 是 0.7370 / 0.7390 / 0.7387 / 0.7277（偏差 0 / 0.5 / 1.0 / 1.5），dev 偏好低偏差，與 public 相反。
- **A 組專用偏差**：2.0 → 0.876、2.5 → 0.872、3.0 → 0.865，**一樣沒有超過全域峰值**，不採用，之後從 notebook 移除。

| 程式輸出的 csv（`output/v7/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission.csv` | — | dev 校準 [0, 0.5, 1.0] | 【未測】 |
| `submission_nocal.csv` | — | 未校準 | 【未測】 |
| `submission_m6_037f63_sel_cal.csv` | — | 同 `submission.csv` | 【未測】 |
| `submission_m6_037f63_sel_nei0.0.csv` | — | NEI +0 | 【未測】 |
| `submission_m6_037f63_sel_nei0.5.csv` | 同名 | NEI +0.5 | 0.856 |
| `submission_m6_037f63_sel_nei1.0.csv` | 同名 | NEI +1.0 | 0.871 |
| `submission_m6_037f63_sel_nei1.5.csv` | 同名 | NEI +1.5（**最終之一**） | **0.881** |
| `submission_m6_037f63_sel_nei2.0.csv` | 同名 | NEI +2.0（**最終之一**） | 0.876 |
| `submission_m6_037f63_sel_nei2.5.csv` | 同名 | NEI +2.5 | 0.870 |
| `submission_m6_037f63_sel_Anei2.0.csv` | 同名 | A 組 +2.0 | 0.876 |
| `submission_m6_037f63_sel_Anei2.5.csv` | 同名 | A 組 +2.5 | 0.872 |
| `submission_m6_037f63_sel_Anei3.0.csv` | 同名 | A 組 +3.0 | 0.865 |

### v8.0 — 三個想法的驗證（全部沒有採用）

- **notebook**：`NLP_HW3_NTHU_115062556v8.0.ipynb`；輸出資料夾 `output/v8/`（A 組偏差 cell 已移除；全域 2.0 / 2.5 也不再輸出）
- 所有比較都對同一批 dev 題目做配對檢定；只有在 NEI 偏差 0 與部署偏差 1.5 兩種設定下都「不變差」才採用。

| 想法 | 做法 | dev 結果 | 結論 |
|---|---|---|---|
| 分類器改讀選擇器的前 4 句（證據一致性） | 6 個成員重新推論（約 56 分鐘，不需重訓） | 配對 @0：+55/−58（z −0.28）；@1.5：+55/−51（z +0.39） | 雜訊，**不採用** |
| 5 折交叉驗證學集成權重 | NLL 目標、權重非負且加總為 1 | @0 z −0.85；@1.5 z +1.26（rr）；學到的權重反而給最弱的 qwen 最大（0.28） | 不穩定，**不採用** |
| refutes 偏差（NEI 固定 1.5） | 偏差 0 / 0.25 / 0.5 / 0.75 | dev CFEVER 0.7277 / 0.7340 / 0.7393 / 0.7430；二折交叉驗證兩折都提升 | dev 有效，但 **public 掉到 0.875**（< 事先定的 0.878 門檻），**不採用** |

- **refutes 偏差的教訓**：它把 A 組被判成 NEI 的比例從 48.8% 拉回 44%，方向正好和 public 偏好的「更多 NEI」相反。**依賴類別比例的參數不能只看 dev**，因為 dev 三類各 1/3，而 test A 組的 NEI 比例更高。
- v8 資料夾的 `nei1.5` 與 `nei1.0` 兩個檔案的 public 與 v7 同名檔一模一樣（0.881 / 0.871），代表 v8 沒有改變主流程的預測（檔案本身的 sha256 比對尚未做）。

| 程式輸出的 csv（`output/v8/`） | 本機保存檔名 | 內容 | public |
|---|---|---|---|
| `submission.csv`、`submission_nocal.csv`、`submission_m6_037f63_sel_cal.csv` | — | 同 v7 的校準 / 未校準 | 【未測】 |
| `submission_m6_037f63_sel_nei0.0.csv` | — | NEI +0 | 【未測】 |
| `submission_m6_037f63_sel_nei0.5.csv` | — | NEI +0.5 | 【未測】 |
| `submission_m6_037f63_sel_nei1.0.csv` | 同名 | NEI +1.0 | 0.871 |
| `submission_m6_037f63_sel_nei1.5.csv` | 同名 | NEI +1.5 | 0.881 |
| `submission_v8_rr_equal_ref0.5_nei1.5.csv` | 同名 | v7 的 nei1.5 + refutes +0.5 | 0.875 |

### v9 — 更強、更多樣的集成成員（成功；最終提交來源，2026-11-16～17）

- **notebook**：最終流程在 `NLP_HW3_NTHU_115062556.ipynb`（重構後；原第 9 節 9a～9f 的實驗在 commit `047ce6e` 的版本裡，9a / v8 / v7 的 csv 寫出程式沒有放進重構版）；輸出 `output/final/`；實驗腳本在本機 `~/NLP-Assignment3/work/`（不在 repo 內）。
- **錯誤分析的出發點**：dev 上「分類錯誤」的損失（約 22 pt）遠大於「證據不完整」（約 4–6 pt）；而且在 dev 上擬合的事後調整（偏差、校準、堆疊）都轉移不到 test。所以 v9 只加「用 train（＋一半 dev）訓練出來的新成員」，集成方式不變。
- **採用規則（事先訂好）**：在 dev NEI 預測比例 0.42 / 0.46 / 0.50 下做配對檢定，三個比例都不變差、至少一個 z ≥ 2；public 不能比 v7 低超過雜訊（約 0.004）。
- **test 組成**：A（id ≤ 31996，3000 題，真正的 CFEVER 題）、B（含「最喜歡」，1000 題，全部 NEI）、C（2000 題，抄自維基第 1 句或只改數字，模型約 99% 對）→ 分數差異幾乎都來自 A。
- **兄弟 claim**：CFEVER 的 claim 成群產生（同一來源句的改寫，id 相鄰）；dev 2000 題 S/R 中 1970 題和 id ±20 內的另一題共用證據頁面。把 2～3 句最相似的兄弟 claim 放進模型輸入（只用 claim 文字與 id，不用標籤），是 v9 最有效的改動。
- **Kaggle GPU**：私人 dataset `yiting224/cfever-hw3-evidence` ＋ 私人 notebook（16GB GPU），訓練程式見 notebook「Classification → Training」code cell 的 `KG_SPECS` / `KG_SCRIPT`。

| 成員 | 訓練 | dev acc | 加入後（相對上一步，三個 NEI 比例） | 結論 |
|---|---|---|---|---|
| `qwen06`：Qwen3-0.6B + LoRA，全部訓練資料 2 epoch | 本機 | 0.7630 | 取代 pilot：+0.63 / +0.53 / +0.33 pt（z 2.7 / 2.3 / 1.3） | 採用 |
| `xlmr_xnli_k6`：XLM-R-large-XNLI，完整微調、證據 6 句、cross-fit | Kaggle | 0.7707（OOF） | +0.10 / +0.30 / +0.30 | 採用 |
| `e330_sib2_k4`：330M-NLI + 2 句兄弟 claim | Kaggle | 0.7697 | +0.63 / +0.10 / +0.67（z 2.5 / 0.4 / 2.6） | 採用 |
| `xlmr_sib2_k4`：XLM-R-XNLI + 2 句兄弟 | Kaggle | 0.7657 | +0.17 / +0.43 / +0.27 | 採用 |
| `e330_sib3_k6`：330M-NLI + 3 句兄弟、證據 6 句 | Kaggle | 0.7720 | 0.00 / −0.03 / +0.17 | 採用（與 xlmr_sib 一起才有效） |
| `mdeberta_sib2_k4`：mDeBERTa-v3-XNLI + 2 句兄弟 | Kaggle | 0.7553 | +0.20 / +0.10 / +0.07 | 採用 |
| `qwen06_sib`：Qwen3-0.6B + 2 句兄弟 | 本機 | **0.7753** | −0.07 / +0.30 / +0.50（z −0.4 / 1.7 / 2.7） | 採用（最終 12 成員） |
| `e330_full_k6`：330M-NLI 完整微調、證據 6 句、cross-fit | Kaggle | 0.7670 | ±0（與現有 330M 一致率 90%） | 不採用 |
| `qwen06_b`：第二顆完整版 Qwen（seed 33） | 本機 | 0.7643 | −0.07 / −0.03 / −0.20（與 qwen06 一致率 91%） | 不採用 |
| `xlmr_sib2_cf`：XLM-R + 兄弟，cross-fit | Kaggle | 0.7767（OOF） | 加入 / 取代皆 ±0.1 | 不採用 |
| `qwen_sib_cf`：Qwen + 兄弟，cross-fit | 本機 | 0.7697（OOF） | 加入 / 取代皆不穩定 | 不採用 |

- **最終 12 成員集成**：dev accuracy 0.7780 → **0.7963**；相對 v7，dev CFEVER 在三個 NEI 比例下 +1.67 / +1.73 / +2.30 pt（z 4.2 / 4.2 / 5.6）。
- **沒有採用的事後調整**：兩視角平均（TTA，部署比例下 dev 沒差別）、群集感知的證據選擇器（OOF Recall@5 0.875 → 0.879，public 差異太小）、群集感知的頁面排序（會動到 C 組）、兄弟 claim 堆疊 / 修正項（dev 上擬合）、refutes 偏差。
- **NEI 比例**：更強的集成最佳點往「更多 NEI」移動，在對齊 v7 偏差 1.75～2.0 的比例形成平台；最終兩份取平台兩端。

| 本帳號提交的 csv | 內容 | public |
|---|---|---|
| `output/v9/sub_v9a_m6tta_nei1.5.csv` | v7 6 成員 + 兩視角平均（TTA），NEI 1.5 | 0.877 |
| `output/v9/sub_v9b_m6tta_sibsel_nei1.5.csv` | v9a + 群集感知選擇器 | 0.878 |
| `output/v9/sub_v9c_m6tta_sibcorr_sibsel_share1.5.csv` | v9b + 兄弟 claim 修正項 | 0.879 |
| `output/v9/sub_v9o_m10_sib_members_share1.5.csv` | 10 成員，NEI 比例對齊 v7 1.5 | 0.886 |
| `output/v9/sub_v9p_m11_all_share1.5.csv` | 11 成員，對齊 1.5 | 0.884 |
| `output/v9/sub_v9q_m12_all_share1.5.csv`（＝ `final/submission_v9_m12_share1.5.csv`） | 12 成員，對齊 1.5 | 0.883 |
| `output/final/submission_v9_m12_share1.75.csv` | 12 成員，對齊 1.75（**最終之一**） | **0.889** |
| `output/final/submission_v9_m12_share2.0.csv` | 12 成員，對齊 2.0（**最終之一**） | 0.888 |
| `output/final/submission_v9_m12_share1.875.csv` | 12 成員，對齊 1.875 | 0.888 |
| `output/final/submission_v9_m12_share1.25.csv`／`2.25` | 12 成員 NEI 比例掃描 | 【待提交】 |

### v10 — private 落差的診斷：NEI 偏差改用 dev 決定（2026-11-23）

- **起因**：11/23 公布的 private 快照只有 **0.8346**（第 243 名），public 0.889（第 19 名）。
- **診斷（只用本帳號的 public 分數、private 快照與 CFEVER 論文的統計表；沒有用任何 test 標籤）**：
  - A 組（id ≤ 31996，3000 題）就是 CFEVER 原本的 test set，論文 Table 3 寫明三類各 1000 題，與 dev 相同 → dev 是 A 組的無偏估計。我們的 share1.75 卻把 A 組 52.3% 判成 NEI。
  - share1.75 的 public + private 總答對題數（2667 + 2504 = 5171）＝ dev 預期（平衡的 A 組在此偏差下 2201 題 + B / C 組約 2970 題）→ 模型表現與 dev 一致，問題在 public / private 的切法。
  - 用 dev 各類別的命中率擬合本帳號 9 個 public 分數（v7 偏差 1.0 / 1.5 / 2.0；v9 share 1.5 / 1.75 / 1.875 / 2.0；v10 偏差 0 / 0.5）：隨機切分的擬合誤差約為最佳解的 100 倍以上；所有好的解都把 **A 組 1000 題 NEI 中的 750–1000 題放在 public**。所以 private 的 A 組幾乎都是 supports / refutes，大的 NEI 偏差把 refutes 改判成 NEI（dev 上 refutes 命中率：偏差 0 為 0.722、偏差 1.981 為 0.531）。
  - 不使用 private 快照的擬合預測 share1.75 的 private ＝ 0.834（實際 0.8346）；不使用 v10 兩個分數的擬合預測 v10 的 public ＝ 0.859 / 0.871（實際 0.862 / 0.872）。
- **修正**：12 成員集成不變，NEI 偏差只用 dev 決定。dev 5 折交叉驗證（依 id 連續切塊）每折都選 0.5（CV CFEVER 0.7597），但 0.5 vs 0 不顯著（+56 / −42，z = 1.41）→ 最終用**不加偏差（0）**，dev CFEVER 0.7550；A 組預測 1046 / 927 / 1027（S / R / NEI），最接近論文的 1000 / 1000 / 1000。
- **刻意不做**：負的 NEI 偏差（擬合預測 private 更高，但那是照推測出的 private 組成調參，且超出 dev 支持的範圍）。
- **最終兩份（2026-11-23 本人已在 Kaggle 勾選）**：`submission_v9_m12_share1.75.csv`（public 0.889）＋ `submission_v10_m12_nei0.0.csv`（public 0.862，擬合預期 private 約 0.884）；private 取兩者較高者。

| 程式輸出的 csv（`output/final/`） | 內容 | public |
|---|---|---|
| `submission_v10_m12_nei0.0.csv` | 12 成員，不加 NEI 偏差（**最終之一**） | 0.862 |
| `submission_v10_m12_nei0.5.csv` | 12 成員，NEI 偏差 0.5（dev 5 折交叉驗證最佳） | 0.872 |

### 定稿 — `NLP_HW3_NTHU_115062556.ipynb`（Report 版）

- **內容**：2026-11-21 起依 `sample_code.ipynb` 的 38 格架構重構（Markdown 全部逐字保留，已用腳本比對；19 個 code cell 依原位置放入 v9 最終 pipeline 的程式，程式內容取自重構前的定稿，沒有改動邏輯）。所有補充說明都只寫在最後一個 cell（Report，題目逐字取自 sample code，只在 `Answer:` 後面填內容）；Generative AI 使用聲明寫在 Report Q5，第一個 code cell 只留兩行指向 Q5 的程式註解（課程要求程式註解也要說明）。重構版不含 v8、9a（兄弟 claim 選擇器）與 v7 csv 的程式；Report 的 Q3 / Q4 已改為對應的說明。
- **狀態**：2026-11-21 以 hw3 kernel（`jupyter nbconvert --execute`，CPU，因 GPU 被其他工作占用）從頭執行並保存輸出，19 個 code cell 無錯誤；`output/final/submission_v9_m12_share{1.25,1.5,1.75,1.875,2.0,2.25}.csv` 重新產生後 sha256 與重構前（2026-11-17 版）逐一相同，所以送出的檔案仍是 0.889 / 0.888 那兩份。dev 指標（notebook 輸出）：頁面 Recall@5 0.9175、證據 Recall@5 0.8780、分類 accuracy 0.7963；dev CFEVER 在部署偏差 1.981 / 2.269（對齊 v7 的 1.75 / 2.0）下為 0.7337 / 0.7143。
- **Report**：private score 欄位留白，公布後補填。

---

## 3. 失敗與沒有採用的實驗彙整

| 實驗 | 版本 | 結果 | 為什麼沒用 / 失敗原因 |
|---|---|---|---|
| 「滿句數 ＝ NEI」捷徑（訓練證據分佈與測試不一致） | v0 | public 0.716 | 三類的訓練證據要用同一個重排器的實際輸出（v1 修正 → 0.833） |
| 超連結第二跳 | 各版都評估（v3.0 之後的 metrics 皆為 `hop_delta = None`） | 最好只多 0.2 pt（Recall@5 0.8615 → 0.8635） | 沒超過採用門檻（> 0.2 pt）；第二跳候選容易擠掉第一跳的正確句子，`hop_delta = None` |
| dev 校準偏差 [0, 0.75, 1.0] | v3.1 | public 0.851 | 低於同模型 NEI +2.0 的 0.873；dev 偏好的 NEI 偏差比 public 低 |
| Qwen3-0.6B LoRA pilot（只用 6000 筆、1 epoch） | v4.0 | 單模型 0.7103；集成約 +0.17 pt | 邊際效果小；保留在集成中，不傷害。完整版未執行 |
| 只對 test A 組加 NEI 偏差 | v5.1、v7 | 0.867 / 0.876 / 0.872 / 0.865 | 沒超過全域偏差峰值 0.881，且要用 id / 「最喜歡」分組，不夠通用 |
| 第二個重排器 + hard negative mining；selector 加 link / rr2 特徵 | v6.0 | OOF Recall@5 全部 0.8780 | 沒有增益 |
| 分類器改讀選擇器證據 | v8.0 | z −0.28 / +0.39 | 雜訊 |
| 交叉驗證學集成權重 | v8.0 | z −0.85 / +1.26 | 不穩定；權重偏愛最弱的 qwen |
| refutes 偏差 | v8.0 | dev +1.5 pt、public −0.6 pt | dev 與 test 類別比例不同 |

**沒有執行過的方向：** MacBERT-large 成員、完整版 Qwen3-0.6B（`qwen06`）、`Erlangshen-MegatronBert-1.3B-NLI`（超過 1B，違反規定，**不能用**）。

---

## 4. 成功的主要設計（Report 的 ablation 可用）

| 元件 | 變化 | 數字 |
|---|---|---|
| 完整 wiki + 學習式頁面排序 | RRF 融合 → HGB 排序 | page Recall@5 0.8535 → 0.9175 |
| 訓練證據與測試一致（v0 → v1） | 修正捷徑 | public 0.716 → 0.833 |
| NLI 預訓練 | `roberta_base` → Erlangshen-110M-NLI | dev acc 0.7227 → 0.7470 |
| 模型大小 | 110M → 330M NLI | 0.7470 → 0.7663 |
| 集成成員數（NEI +1.5） | 3 → 5 → 5+選擇器 → 6+選擇器 | public 0.861 → 0.875 → 0.878 → 0.881 |
| 證據句選擇器 | 重排器 → +選擇器 | evidence Recall@5 0.8615 → 0.8780 |
| NEI 偏差（v7，public） | 0.5 / 1.0 / 1.5 / 2.0 / 2.5 | 0.856 / 0.871 / 0.881 / 0.876 / 0.870 |

---

## 5. 所有「有 public 分數」的檔案（由高到低）

| public | 檔案 | 版本 |
|---|---|---|
| **0.889** | `final/submission_v9_m12_share1.75.csv`（12 成員，**最終之一**） | v9 |
| 0.888 | `final/submission_v9_m12_share2.0.csv`（12 成員；11/23 前的最終之一） | v9 |
| 0.888 | `final/submission_v9_m12_share1.875.csv` | v9 |
| 0.872 | `final/submission_v10_m12_nei0.5.csv`（12 成員，NEI 偏差 0.5） | v10 |
| 0.862 | `final/submission_v10_m12_nei0.0.csv`（12 成員，不加偏差，**最終之一**） | v10 |
| 0.886 | `sub_v9o_m10_sib_members_share1.5.csv`（10 成員） | v9 |
| 0.884 | `sub_v9p_m11_all_share1.5.csv`（11 成員） | v9 |
| 0.883 | `sub_v9q_m12_all_share1.5.csv`（12 成員 ＝ `final/submission_v9_m12_share1.5.csv`） | v9 |
| 0.881 | `submission_m6_037f63_sel_nei1.5.csv`（v7；v8 同名檔也是 0.881） | v7.0 / v8.0 |
| 0.879 | `sub_v9c_m6tta_sibcorr_sibsel_share1.5.csv` | v9 |
| 0.878 | `sub_v9b_m6tta_sibsel_nei1.5.csv` | v9 |
| 0.877 | `sub_v9a_m6tta_nei1.5.csv` | v9 |
| 0.878 | `submission_m5_d13d8f_sel_nei1.5.csv`（v5.0 與 v6.0 同分） | v5.0 / v6.0 |
| 0.876 | `submission_m6_037f63_sel_nei2.0.csv` | v7.0 |
| 0.876 | `submission_m6_037f63_sel_Anei2.0.csv` | v7.0 |
| 0.875 | `submission_v8_rr_equal_ref0.5_nei1.5.csv` | v8.0 |
| 0.875 | `submission_m5_d13d8f_nei1.5.csv` | v4.0 |
| 0.874 | `submission_m5_d13d8f_sel_nei2.0.csv` | v5.0 + v5.1 cell |
| 0.873 | `submission_nei2.0v3.1.csv` | v3.1 |
| 0.872 | `submission_m6_037f63_sel_Anei2.5.csv` | v7.0 |
| 0.871 | `submission_m6_037f63_sel_nei1.0.csv`（v7；v8 同名檔也是 0.871） | v7.0 / v8.0 |
| 0.870 | `submission_m6_037f63_sel_nei2.5.csv` | v7.0 |
| 0.867 | `submission_m5_d13d8f_sel_Anei2.5.csv` | v5.0 + v5.1 cell |
| 0.865 | `submission_m6_037f63_sel_Anei3.0.csv` | v7.0 |
| 0.861 | `submission_nei1.5v3.csv` | v3.0 |
| 0.856 | `submission_m6_037f63_sel_nei0.5.csv` | v7.0 |
| 0.854 | `submission_nei3.0v3.1.csv` | v3.1 |
| 0.852 | `submission_nei1.0v3.csv` | v3.0 |
| 0.851 | `submissionv3.1.csv` | v3.1 |
| 0.844 | `submission_nei0.5v3.csv` | v3.0 |
| 0.838 | `submission.csv`（v5.0，NEI 偏差 0） | v5.0 |
| 0.833 | `submissionv1.csv` | v1 |
| 0.824 | `submissionv3.csv` | v3.0 |
| 0.810 | `submissionv2.csv` | v2 |
| 0.809 | `submission_nocalv1.csv` | v1 |
| 0.716 | `submissionv0.csv` | v0 |

（第一名 public 約 0.890，2026-11-09 時；2026-11-17 時第一名 0.902、第五名 0.898。）

---

## 6. 教訓

1. **訓練輸入必須和測試輸入同分佈**：v0 的捷徑 bug 讓 public 差了 11.7 個百分點。
2. **集成的增益是真的，照 public 挑檔不是**：成員數、NLI 預訓練、選擇器的提升都能在 dev 上用配對檢定驗證；NEI 偏差 1.5 還是 2.0 的差距小於一個標準誤，所以最終兩份各留一份對沖。
3. **類別比例相關的參數不能只看 dev**：dev 三類各 1/3，test A 組的 NEI 比例更高（偏差 1.5 時 A 組被判成 NEI 的比例 48.8%）。dev 校準出來的 NEI 偏差（+1.0）、refutes 偏差（+0.5～0.75）都沒能轉移到 public。
4. **每次改動都用配對檢定（好 / 壞題數）判斷是不是雜訊**，而不是只看分數的小數點。
5. **在 dev 上擬合的事後調整一律不可靠**（v9）：兩視角平均、兄弟 claim 堆疊、refutes 偏差在 dev 都有效，public 卻沒有進步；同樣的兄弟 claim 資訊改成「放進模型輸入、用 train 訓練」才同時在 dev 與 public 進步。
6. **集成需要「錯誤不同」的成員**（v9）：同一個模型只換 seed（一致率 90% 以上）幾乎沒有幫助；不同的模型家族（XLM-R、mDeBERTa、Qwen）或不同的輸入資訊（兄弟 claim）才有幫助。
7. **NEI 最佳比例會隨集成強度改變**（v9）：v7 的最佳點是偏差 1.5，12 成員集成在對齊 1.75～2.0 的比例形成平台，所以換集成後要重新掃描，而不是沿用舊的偏差值。
8. ~~private 分數公布前不下結論；最終兩份取平台兩端~~ → **public 不是 test 的隨機樣本**（v10）：public 的 A 組 NEI 題特別多，照 public 調的 NEI 偏差在 private 掉到 0.8346。當資料集論文已說明 dev 與 test 的類別分佈相同時，類別比例相關的參數只能用 dev 決定；public 偏好與 dev 矛盾時，應視為 public 子集的性質，而不是 test 的性質。（第 3 點的「test A 組的 NEI 比例更高」是錯的推論。）
