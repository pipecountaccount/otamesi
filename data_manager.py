import csv
import io
import json
import re
from pathlib import Path

# -------------------------------------------------
# 1. 保存先の設定
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"  # 旧データとの互換性のため残す
CSV_PATH = DATA_DIR / "combined.csv"
LOCATIONS_PATH = DATA_DIR / "locations.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# 2. ファイル名の安全化
# -------------------------------------------------
def safe_filename_part(value, default="unknown"):
    value = str(value).strip()
    value = re.sub(r'[\\/:*?"<>|]', "_", value)
    value = value.strip(". ")
    return value if value else default


# -------------------------------------------------
# 3. 場所データの読み込み・保存
# -------------------------------------------------
def load_locations():
    if not LOCATIONS_PATH.exists():
        return []
    try:
        with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
            locations = json.load(f)
        if isinstance(locations, list):
            return [str(item).strip() for item in locations if str(item).strip()]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_locations(locations):
    unique_locations = []
    seen = set()
    for item in locations:
        name = str(item).strip()
        if name and name not in seen:
            unique_locations.append(name)
            seen.add(name)

    with open(LOCATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_locations, f, ensure_ascii=False, indent=4)


def add_location(location):
    location = str(location).strip()
    if not location:
        return
    locations = load_locations()
    if location not in locations:
        locations.append(location)
        save_locations(locations)


def import_locations_from_csv(file_bytes):
    """CSVの「場所」列、なければ1列目から場所を読み込み、重複を除いて登録する。"""
    text = None
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            text = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("文字コードを判定できませんでした。UTF-8またはShift-JISのCSVを使用してください。")

    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel

    rows = list(csv.reader(io.StringIO(text), dialect))
    if not rows:
        return 0, 0

    # ヘッダーに「場所」があればその列、なければ先頭列を使用
    header = [cell.strip().replace("\ufeff", "") for cell in rows[0]]
    location_col = next(
        (i for i, name in enumerate(header) if name in ("場所", "保管場所", "location", "Location")),
        None,
    )
    start_row = 1 if location_col is not None else 0
    if location_col is None:
        location_col = 0

    imported = []
    for row in rows[start_row:]:
        if len(row) > location_col:
            value = row[location_col].strip()
            if value and value not in imported:
                imported.append(value)

    before = load_locations()
    before_set = set(before)
    merged = before + [name for name in imported if name not in before_set]
    added_count = len(merged) - len(before)
    save_locations(merged)
    return added_count, len(imported)


# -------------------------------------------------
# 4. 判定結果の読み込み・保存
# -------------------------------------------------
FIELDNAMES = [
    "日時",
    "グループ",
    "場所",
    "AI検出数",
    "手動追加数",
    "最終合計",
    "備考",
]


def load_results():
    if not CSV_PATH.exists():
        return []
    try:
        with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _write_results(records):
    """既存レコードをCSVに書き直す。旧CSVに画像ファイル名列があっても読み込み可能。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = FIELDNAMES
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def append_result(record):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0

    # 旧バージョンのCSVに「画像ファイル名」列がある場合も、新形式へ整理する
    if file_exists:
        existing = load_results()
        _write_results(existing + [record])
    else:
        with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerow({key: record.get(key, "") for key in FIELDNAMES})


def delete_result(index):
    """CSV内の指定位置のレコードを削除する。"""
    records = load_results()
    if index < 0 or index >= len(records):
        return False
    records.pop(index)
    _write_results(records)
    return True


def delete_results(indices):
    """CSV内の複数レコードを一括削除する。"""
    records = load_results()
    valid = {int(i) for i in indices if 0 <= int(i) < len(records)}
    if not valid:
        return False
    remaining = [record for i, record in enumerate(records) if i not in valid]
    _write_results(remaining)
    return True


def delete_locations(names):
    """登録済み場所から指定した複数の場所を削除し、削除件数を返す。"""
    targets = {str(name).strip() for name in names if str(name).strip()}
    before = load_locations()
    remaining = [name for name in before if name not in targets]
    removed = len(before) - len(remaining)
    if removed:
        save_locations(remaining)
    return removed
