# NLP HW3 — CFEVER 中文事實查核：實驗紀錄

學號 115062556　｜　Kaggle username 17568160　｜　競賽：`nthu-nlp-2026-assignment-3-fact-verification`（2026-11-04 09:00 ～ 2026-11-23 12:00）

這份文件記錄**所有實驗（成功與失敗）**、每個實驗對應的 notebook 檔名、每個 notebook 輸出了哪些 `.csv`，以及各 `.csv` 在 Kaggle 的 public score。
**從未上傳 Kaggle 的 `.csv`，public score 一律填【未測】。**

- 作業規定與提醒請看 `NOTE.md`。
- 分數＝Kaggle FEVER score（與 notebook 內的 CFEVER 相同）。dev 指標都在 dev 3000 題（supports / refutes / NEI 各 1000）上計算。
- 完整的 Report 草稿存在專案 `claude/hw3_report_draft.md`。

---

## 0. 一眼看完

**目前最終提交（Kaggle 上要勾選的兩份，截止前手動選）：**

| 順序 | 檔案 | 產生它的 notebook | public | NEI 偏差 |
|---|---|---|---|---|
| 1 | `output/v7/submission_m6_037f63_sel_nei1.5.csv` | `NLP_HW3_NTHU_115062556v7.0.ipynb` | **0.881** | +1.5 |
| 2 | `output/v7/submission_m6_037f63_sel_nei2.0.csv` | `NLP_HW3_NTHU_115062556v7.0.ipynb` | 0.876 | +2.0 |

定稿 notebook：`NLP_HW3_NTHU_115062556.ipynb`（以 v8.0 為底 + Report；輸出到 `output/final/`，**尚未執行，其 CSV 皆未送過 Kaggle**）。

**最終 pipeline（v7.0 ＝ 定稿的主流程）：**
1. 頁面檢索：標題子字串 + BM25（字元 bigram）+ 稠密檢索（`bge-small-zh-v1.5`）→ HistGradientBoosting 學習式排序（完整 wiki 119 萬頁），前 6 頁。
2. 證據檢索：bigram 粗篩 32 句 → MacBERT cross-encoder 重排 → 證據句選擇器（HGB，dev 5 折 OOF）→ 前 5 句。
3. 分類：6 成員集成（macbert、roberta、Erlangshen-110M-NLI、Erlangshen-330M-NLI × 2 種子、Qwen3-0.6B LoRA），claim + 前 4 句，log-softmax 等權平均 + NEI 偏差。

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

### 定稿 — `NLP_HW3_NTHU_115062556.ipynb`（Report 版）

- **內容**：以 v8.0 為底；更新開頭說明與 GenAI 聲明；輸出資料夾 `output/final/`；NEI 偏差掃描擴充到 0.0～2.5（會產生最終兩份提交的檔案）；v8 一節標明為負面結果；最後一個 cell 是 Report（題目逐字取自 sample code，只在 `Answer:` 後面填內容）。
- **狀態**：已用模擬環境跑過所有程式 cell 的流程；**真實模型與 GPU 尚未執行**，所以 notebook 內沒有輸出，下列 CSV 全部【未測】。
- 預期 `final/submission_m6_037f63_sel_nei1.5.csv` / `nei2.0.csv` 與 v7 的同名檔內容相同，需要用 `sha256sum` 驗證（驗證前以 v7 的檔案為準）。

| 程式輸出的 csv（`output/final/`） | 內容 | public |
|---|---|---|
| `submission.csv`、`submission_nocal.csv`、`submission_m6_037f63_sel_cal.csv` | 校準 / 未校準 | 【未測】 |
| `submission_m6_037f63_sel_nei0.0.csv` … `_nei2.5.csv`（0.0、0.5、1.0、1.5、2.0、2.5） | 全域 NEI 偏差 | 【未測】 |
| `submission_v8_*.csv`（v8 一節的 refutes 偏差檔等） | v8 實驗檔 | 【未測】 |

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
| **0.881** | `submission_m6_037f63_sel_nei1.5.csv`（v7；v8 同名檔也是 0.881） | v7.0 / v8.0 |
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

（第一名 public 約 0.890，2026-11-09 時。）

---

## 6. 教訓

1. **訓練輸入必須和測試輸入同分佈**：v0 的捷徑 bug 讓 public 差了 11.7 個百分點。
2. **集成的增益是真的，照 public 挑檔不是**：成員數、NLI 預訓練、選擇器的提升都能在 dev 上用配對檢定驗證；NEI 偏差 1.5 還是 2.0 的差距小於一個標準誤，所以最終兩份各留一份對沖。
3. **類別比例相關的參數不能只看 dev**：dev 三類各 1/3，test A 組的 NEI 比例更高（偏差 1.5 時 A 組被判成 NEI 的比例 48.8%）。dev 校準出來的 NEI 偏差（+1.0）、refutes 偏差（+0.5～0.75）都沒能轉移到 public。
4. **每次改動都用配對檢定（好 / 壞題數）判斷是不是雜訊**，而不是只看分數的小數點。
5. private 分數粗估 0.87～0.88，公布前不下結論。
