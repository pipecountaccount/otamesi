
import csv
import json
import re
from pathlib import Path

# -------------------------------------------------
# 1. 保存先の設定
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"

CSV_PATH = DATA_DIR / "combined.csv"
LOCATIONS_PATH = DATA_DIR / "locations.json"

# 保存用フォルダがなければ作成
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# 2. ファイル名の安全化
# -------------------------------------------------

def safe_filename_part(value, default="unknown"):
    """ファイル名に使用できない文字を置き換える"""

    value = str(value).strip()

    value = re.sub(
        r'[\\/:*?"<>|]',
        "_",
        value
    )

    value = value.strip(". ")

    if not value:
        return default

    return value


# -------------------------------------------------
# 3. 場所データの読み込み
# -------------------------------------------------

def load_locations():
    """保存されている場所の一覧を読み込む"""

    if not LOCATIONS_PATH.exists():
        return []

    try:
        with open(
            LOCATIONS_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            locations = json.load(f)

        if isinstance(locations, list):
            return locations

        return []

    except (json.JSONDecodeError, OSError):
        return []


# -------------------------------------------------
# 4. 場所データの追加
# -------------------------------------------------

def add_location(location):
    """新しい場所を保存する（重複登録しない）"""

    location = str(location).strip()

    if not location:
        return

    locations = load_locations()

    if location not in locations:
        locations.append(location)

        with open(
            LOCATIONS_PATH,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                locations,
                f,
                ensure_ascii=False,
                indent=4
            )


# -------------------------------------------------
# 5. 判定結果の読み込み
# -------------------------------------------------

def load_results():
    """CSVから過去の判定結果を読み込む"""

    if not CSV_PATH.exists():
        return []

    try:
        with open(
            CSV_PATH,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.DictReader(f)
            return list(reader)

    except OSError:
        return []


# -------------------------------------------------
# 6. 判定結果の追加
# -------------------------------------------------

def append_result(record):
    """判定結果をCSVに追加する"""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = CSV_PATH.exists()

    fieldnames = [
        "日時",
        "グループ",
        "場所",
        "AI検出数",
        "手動追加数",
        "最終合計",
        "備考",
        "画像ファイル名"
    ]

    with open(
        CSV_PATH,
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        # 新規ファイルの場合は見出しを追加
        if not file_exists or CSV_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(record)
