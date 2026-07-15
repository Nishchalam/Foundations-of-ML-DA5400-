# manual_labeler.py
import os, glob, csv
from pathlib import Path

TEST_DIR = "src/data/test"
OUT = os.path.join(TEST_DIR, "test_labels.csv")
Path(TEST_DIR).mkdir(exist_ok=True)

files = sorted(glob.glob(os.path.join(TEST_DIR, "*.txt")))
if not files:
    print("No files in ./test. Create test/*.txt first."); raise SystemExit(1)

rows = []
print("Label: [s]pam / [h]am / [k] skip (enter to repeat)")
for f in files:
    print("\n----")
    print(f"File: {os.path.basename(f)}")
    print("---------------------------------------")
    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read(2000)   # show up to first 2k chars
        print(text)
    cmd = input("\nLabel (s/h/k) > ").strip().lower()
    if cmd == 's':
        rows.append((os.path.basename(f), "+1"))
    elif cmd == 'h':
        rows.append((os.path.basename(f), "0"))
    elif cmd == 'k':
        print("Skipped")
    else:
        print("Invalid. Skipped.")
# write CSV
with open(OUT, "w", newline='', encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["filename","label"])
    writer.writerows(rows)
print("Wrote:", OUT)
