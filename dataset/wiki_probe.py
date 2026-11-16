"""wiki_probe.py - 讀取 wiki_pages.jsonl，輸出統計報告與一個小型抽樣檔。

只用 Python 標準函式庫，不需要 GPU。用法（在放有資料的資料夾執行）：
    python wiki_probe.py --wiki wiki_pages.jsonl --train train.jsonl --dev dev.jsonl

產生：
    wiki_stats.txt    統計報告（請把內容貼回對話即可）
    wiki_sample.jsonl 小型抽樣檔（dev 的證據頁面 + 部分 train 證據頁面 + 隨機頁面），可上傳
"""
import argparse
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設編碼可能無法顯示中文

parser = argparse.ArgumentParser()
parser.add_argument("--wiki", default="wiki_pages.jsonl")
parser.add_argument("--train", default="train.jsonl")
parser.add_argument("--dev", default="dev.jsonl")
parser.add_argument("--out_dir", default=".")
parser.add_argument("--random_pages", type=int, default=2000)
parser.add_argument("--train_gold_sample", type=int, default=300)
parser.add_argument("--probe_claims", type=int, default=500)
args = parser.parse_args()

random.seed(2026)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
report = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    report.append(line)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gold_pages(rows):
    pages = set()
    for row in rows:
        if str(row["label"]).upper() == "NOT ENOUGH INFO":
            continue
        for group in row["evidence"]:
            for item in group:
                pages.add(item[2])
    return pages


train_rows = read_jsonl(args.train)
dev_rows = read_jsonl(args.dev)
train_gold = gold_pages(train_rows)
dev_gold = gold_pages(dev_rows)
train_gold_pick = set(random.sample(sorted(train_gold), min(args.train_gold_sample, len(train_gold))))
must_keep = dev_gold | train_gold_pick

# ---- 單次掃描 wiki：統計、保留指定頁面、水庫抽樣隨機頁面 ----
titles = set()
page_count = 0
total_text_chars = 0
total_sentences = 0
text_lengths = []
kept = {}
reservoir = []
first_page = None
keys_seen = set()

with open(args.wiki, encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue
        page = json.loads(line)
        page_count += 1
        keys_seen.update(page.keys())
        title = page.get("id", "")
        titles.add(title)
        text = page.get("text", "") or ""
        total_text_chars += len(text)
        text_lengths.append(len(text))
        total_sentences += page.get("lines", "").count("\n") + 1 if page.get("lines") else 0
        if first_page is None:
            first_page = page
        if title in must_keep:
            kept[title] = page
        else:
            if len(reservoir) < args.random_pages:
                reservoir.append(page)
            else:
                j = random.randint(0, page_count - 1)
                if j < args.random_pages:
                    reservoir[j] = page
        if page_count % 200000 == 0:
            print("  已讀取頁面：", page_count)

text_lengths.sort()


def pct(p):
    return text_lengths[min(len(text_lengths) - 1, int(len(text_lengths) * p))] if text_lengths else 0


log("=== Wiki 基本統計 ===")
log("頁面數:", page_count)
log("檔案大小(MB):", round(Path(args.wiki).stat().st_size / 1e6, 1))
log("欄位:", sorted(keys_seen))
log("text 平均字數:", round(total_text_chars / max(page_count, 1), 1))
log("text 字數 中位數/P90/P99/最大:", pct(0.5), pct(0.9), pct(0.99), text_lengths[-1] if text_lengths else 0)
log("句子總數(以 lines 換行估計):", total_sentences)

log("\n=== 範例頁面（第一頁）===")
if first_page:
    log("id:", first_page.get("id"))
    log("text[:200]:", repr(first_page.get("text", "")[:200]))
    log("lines[:300]:", repr(first_page.get("lines", "")[:300]))

log("\n=== 證據頁面覆蓋率（wiki 是否包含標準答案頁面）===")
log("train 證據頁面: %d / %d 在 wiki 中" % (len(train_gold & titles), len(train_gold)))
log("dev   證據頁面: %d / %d 在 wiki 中" % (len(dev_gold & titles), len(dev_gold)))
log("dev 頁面與 train 頁面重疊比例:", round(len(dev_gold & train_gold) / max(len(dev_gold), 1), 4))

# ---- 標題子字串比對：候選數量是否爆炸 ----
log("\n=== 標題比對探測（claim 內有哪些 wiki 標題以子字串出現）===")
sr_rows = [r for r in dev_rows if str(r["label"]).upper() != "NOT ENOUGH INFO"]
sr_rows = random.sample(sr_rows, min(args.probe_claims, len(sr_rows)))
match_counts = []
hit_all = 0
single_char = 0
for row in sr_rows:
    claim = row["claim"]
    found = set()
    for i in range(len(claim)):
        for j in range(i + 1, min(len(claim), i + 20) + 1):
            sub = claim[i:j]
            if sub in titles:
                found.add(sub)
    match_counts.append(len(found))
    single_char += sum(1 for t in found if len(t) == 1)
    best_group = min(row["evidence"], key=len)
    if all(item[2] in found for item in best_group):
        hit_all += 1
match_counts.sort()
n = max(len(sr_rows), 1)
log("探測題數:", len(sr_rows))
log("最佳證據組的所有頁面標題都在比對結果中:", hit_all, "/", len(sr_rows), "=", round(hit_all / n, 4))
log("每題比對到的標題數 平均/中位/P90/最大:",
    round(sum(match_counts) / n, 1), match_counts[len(match_counts) // 2],
    match_counts[int(len(match_counts) * 0.9)], match_counts[-1])
log("單字標題平均每題出現數:", round(single_char / n, 2))

# ---- 輸出抽樣檔 ----
sample_path = out_dir / "wiki_sample.jsonl"
with open(sample_path, "w", encoding="utf-8") as file:
    for page in list(kept.values()) + reservoir:
        file.write(json.dumps(page, ensure_ascii=False) + "\n")
log("\n抽樣檔:", sample_path, "頁數:", len(kept) + len(reservoir),
    "大小(MB):", round(sample_path.stat().st_size / 1e6, 2))

with open(out_dir / "wiki_stats.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(report) + "\n")
print("\n完成。請把 wiki_stats.txt 的內容貼給我；wiki_sample.jsonl 若小於數十 MB 也可一併上傳。")
