import base64
import hashlib
import io
import re
import zipfile
from datetime import datetime
from html import escape

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates

from data_manager import (
    add_location,
    append_result,
    delete_result,
    delete_results,
    delete_locations,
    import_locations_from_csv,
    load_locations,
    load_results,
    safe_filename_part,
    CSV_PATH,
    DATA_DIR,
)

# -------------------------------------------------
# 1. 初期設定とモデルロード
# -------------------------------------------------
st.set_page_config(page_title="PINEAPPLE Pipe Counter", layout="wide")


@st.cache_resource
def load_model():
    return YOLO("last_openvino_model")


model = load_model()
TARGET_SIZE = 960

# -------------------------------------------------
# 2. Session State
# -------------------------------------------------
DEFAULTS = {
    "phase": "upload",
    "mask_rects": [],
    "temp_point": None,
    "last_processed_click": None,
    "points": [],
    "auto_boxes": None,
    "current_upload_id": None,
    "saved_zip_data": None,
    "saved_filename": None,
    "saved_message": None,
    "location_import_message": None,
    "zip_downloaded": False,
    "flash_message": None,
    "flash_kind": "success",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------------------------------------
# 3. 便利関数
# -------------------------------------------------
def apply_masks(img, rects):
    """画像に黒塗りのマスクを適用する"""
    draw = ImageDraw.Draw(img)
    for rect in rects:
        draw.rectangle(rect, fill="black")
    return img


def resize_with_aspect_ratio(img, target_max_side):
    """アスペクト比を維持してリサイズする"""
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = target_max_side / max(w, h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def reset_current_image():
    """次の画像に進むため、画像ごとの状態を初期化する"""
    st.session_state.phase = "upload"
    st.session_state.mask_rects = []
    st.session_state.temp_point = None
    st.session_state.last_processed_click = None
    st.session_state.points = []
    st.session_state.auto_boxes = None
    st.session_state.current_upload_id = None
    st.session_state.saved_zip_data = None
    st.session_state.saved_filename = None
    st.session_state.saved_message = None
    st.session_state.zip_downloaded = False


def set_flash(message, kind="success"):
    st.session_state.flash_message = message
    st.session_state.flash_kind = kind


def show_flash():
    message = st.session_state.get("flash_message")
    if message:
        kind = st.session_state.get("flash_kind", "success")
        if kind == "error":
            st.error(message)
        elif kind == "warning":
            st.warning(message)
        else:
            st.success(message)
        st.session_state.flash_message = None


def clear_detection_data_after_download():
    """ZIPのダウンロード操作後、検出結果だけを消去し、場所一覧は保持する。"""
    if CSV_PATH.exists():
        CSV_PATH.unlink()
    reports_dir = DATA_DIR / "reports"
    if reports_dir.exists():
        for report_file in reports_dir.glob("*.html"):
            report_file.unlink()
    st.session_state.zip_downloaded = True
    set_flash("ZIPのダウンロード操作を受け付けました。検出結果データを消去しました（過去の場所は保持しています）。")


def make_html_report(record, image_bytes=None):
    """画像をHTML内に埋め込んだ、単体で閲覧できるレポートを作成する"""
    image_section = ""
    if image_bytes:
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        image_section = (
            '<h2>検出結果の画像</h2>'
            f'<img src="data:image/png;base64,{encoded_image}" alt="検出結果の画像">'
        )
    else:
        image_section = "<p>この記録には検出結果画像が保存されていません。</p>"
    rows = [
        ("日時", record.get("日時", "")),
        ("グループ", record.get("グループ", "")),
        ("場所", record.get("場所", "")),
        ("AI検出数", record.get("AI検出数", "")),
        ("手動追加数", record.get("手動追加数", "")),
        ("最終合計", f'{record.get("最終合計", "")} 本'),
        ("備考", record.get("備考", "")),
    ]
    table_rows = "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in rows
    )
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PINEAPPLE 検出結果レポート</title>
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 30px auto; padding: 0 18px; color: #222; }}
h1 {{ border-bottom: 3px solid #65a30d; padding-bottom: 10px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ccc; padding: 10px; text-align: left; }}
th {{ width: 30%; background: #f3f4f6; }}
img {{ display: block; max-width: 100%; height: auto; margin: 20px auto; border: 1px solid #ddd; }}
.total {{ font-size: 1.5rem; font-weight: bold; }}
</style>
</head>
<body>
<h1>PINEAPPLE 検出結果レポート</h1>
<p class="total">最終合計：{escape(str(record.get("最終合計", "")))} 本</p>
<h2>検出情報</h2>
<table>{table_rows}</table>
{image_section}
</body>
</html>"""


def export_basename(record):
    """日付・時刻・グループ名を使ったファイル名を作成する。"""
    raw_datetime = str(record.get("日時", "")).strip()
    # 新形式: YYYYMMDD_HHMMSS -> YYYY-MM-DD_HH-MM-SS
    if len(raw_datetime) >= 15 and raw_datetime[:8].isdigit():
        date_part = f"{raw_datetime[:4]}-{raw_datetime[4:6]}-{raw_datetime[6:8]}"
        time_part = raw_datetime[9:15].replace(":", "")
        if time_part.isdigit():
            time_part = f"{time_part[:2]}-{time_part[2:4]}-{time_part[4:6]}"
        else:
            time_part = safe_filename_part(time_part, "no_time")
    else:
        try:
            parsed = datetime.fromisoformat(raw_datetime)
            date_part = parsed.strftime("%Y-%m-%d")
            time_part = parsed.strftime("%H-%M-%S")
        except ValueError:
            date_part = safe_filename_part(raw_datetime, "no_date")
            time_part = "no_time"
    group_part = safe_filename_part(record.get("グループ", ""), "no_group")
    return f"{date_part}_{time_part}_{group_part}"


def make_all_html_report(records):
    """保存済みの全レコードを一覧表示するHTMLレポートを作成する。"""
    sections = []
    reports_dir = DATA_DIR / "reports"
    for number, record in enumerate(records, start=1):
        rows = [
            ("日時", record.get("日時", "")),
            ("グループ", record.get("グループ", "")),
            ("場所", record.get("場所", "")),
            ("AI検出数", record.get("AI検出数", "")),
            ("手動追加数", record.get("手動追加数", "")),
            ("最終合計", f'{record.get("最終合計", "")} 本'),
            ("備考", record.get("備考", "")),
        ]
        table_rows = "\n".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in rows
        )

        # 保存済みの個別HTMLから画像を引き継ぐ（旧ファイル名にも対応）
        image_html = ""
        legacy_dt = safe_filename_part(record.get("日時", ""), "no_date")
        legacy_loc = safe_filename_part(record.get("場所", ""), "no_location")
        current_base = export_basename(record)
        previous_base = current_base.rsplit("_", 1)[0]
        candidates = [
            reports_dir / f"{current_base}.html",
            reports_dir / f"{legacy_dt}_{legacy_loc}.html",
            reports_dir / f"{previous_base}.html",
        ]
        for report_path in candidates:
            if report_path.exists():
                old_html = report_path.read_text(encoding="utf-8")
                match = re.search(r'src="data:image/png;base64,([^"]+)"', old_html)
                if match:
                    image_html = (
                        '<h3>検出結果の画像</h3>'
                        f'<img alt="検出結果の画像" src="data:image/png;base64,{match.group(1)}">'
                    )
                    break
        sections.append(
            f'<section><h2>記録 {number}</h2><table>{table_rows}</table>{image_html}</section>'
        )

    body = "\n".join(sections) if sections else "<p>保存データはありません。</p>"
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PINEAPPLE 保存データ一覧</title>
<style>
body {{ font-family: sans-serif; max-width: 1000px; margin: 30px auto; padding: 0 18px; color: #222; }}
h1 {{ border-bottom: 3px solid #65a30d; padding-bottom: 10px; }}
section {{ margin: 28px 0; page-break-inside: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #ccc; padding: 10px; text-align: left; }}
th {{ width: 30%; background: #f3f4f6; }}
img {{ display: block; max-width: 100%; height: auto; margin: 20px auto; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>PINEAPPLE 検出結果一覧</h1>
<p>保存件数：{len(records)} 件</p>
{body}
</body>
</html>"""


def create_export_zip(record, html_text):
    """統合CSVと全保存データのHTMLをZIPにまとめる。"""
    zip_buffer = io.BytesIO()
    csv_bytes = CSV_PATH.read_bytes() if CSV_PATH.exists() else b""
    base_name = export_basename(record)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{base_name}.csv", csv_bytes)
        archive.writestr(f"{base_name}.html", html_text.encode("utf-8"))
    return zip_buffer.getvalue()


def create_saved_record_export(record):
    """統合CSVと、保存済み全レコードのHTMLをダウンロードする。"""
    return create_export_zip(record, make_all_html_report(load_results()))


# 4. 画面切り替え
# -------------------------------------------------
st.title("PINEAPPLE 🍍")
show_flash()
page = st.sidebar.radio("画面", ["画像を判定する", "保存データを確認する"])


# -------------------------------------------------
# 5. 保存データ確認画面
# -------------------------------------------------
if page == "保存データを確認する":
    st.header("保存データの確認")

    st.subheader("場所をCSVから一括登録")
    st.caption("CSVの「場所」列、または1列目に場所を記入してください。")
    locations_csv = st.file_uploader(
        "場所一覧のCSVファイル",
        type=["csv"],
        key="locations_csv_uploader",
    )
    if locations_csv is not None:
        if st.button("CSVの場所を登録する", key="import_locations_button"):
            try:
                added, total = import_locations_from_csv(locations_csv.getvalue())
                set_flash(f"{total}件の場所を読み込みました。新規登録は{added}件です。")
                st.rerun()
            except Exception as exc:
                st.error(f"CSVを読み込めませんでした: {exc}")

    st.divider()
    st.subheader("過去に登録した場所の管理")
    saved_locations = load_locations()
    if saved_locations:
        selected_locations = st.multiselect(
            "削除する場所を複数選択してください",
            options=saved_locations,
            key="locations_to_delete",
        )
        if st.button("🗑️ 選択した場所をまとめて削除", disabled=not selected_locations):
            removed = delete_locations(selected_locations)
            st.session_state.pop("locations_to_delete", None)
            set_flash(f"過去の場所を{removed}件削除しました。")
            st.rerun()
    else:
        st.info("登録済みの場所はありません。")

    st.divider()
    records = load_results()

    if not records:
        st.info("まだ保存データがありません。画像を判定して保存してください。")
    else:
        indexed_records = list(enumerate(records))
        indexed_records.reverse()
        labels = [
            f'{r.get("日時", "")} | {r.get("グループ", "")} | '
            f'{r.get("場所", "")} | {r.get("最終合計", "")}本'
            for _, r in indexed_records
        ]
        st.subheader("保存データ一覧")
        st.caption(f"保存されている全 {len(records)} 件を確認できます。")
        selected = st.selectbox(
            "詳細を確認するデータ",
            range(len(indexed_records)),
            format_func=lambda i: labels[i],
        )
        original_index, record = indexed_records[selected]
        st.subheader("選択したデータの内容")
        st.dataframe([record], use_container_width=True)

        export_data = create_saved_record_export(record)
        export_name = export_basename(record)
        st.download_button(
            "⬇️ 全保存データのCSVとHTMLをダウンロード",
            data=export_data,
            file_name=f"{export_name}_全データ.zip",
            mime="application/zip",
            type="primary",
            on_click="ignore",
        )

        st.caption("ダウンロード後、必要であれば下のボタンからアプリ内の検出結果を削除できます。")
        if st.button("🗑️ ダウンロード後に検出結果をすべて削除", key="clear_after_saved_page"):
            clear_detection_data_after_download()
            st.rerun()

        st.divider()
        st.subheader("保存データの一括削除")
        deletion_choices = {
            f'{r.get("日時", "")} | {r.get("グループ", "")} | '
            f'{r.get("場所", "")} | {r.get("最終合計", "")}本（CSV行 {i + 1}）': i
            for i, r in enumerate(records)
        }
        selected_delete_labels = st.multiselect(
            "削除するデータを複数選択してください",
            options=list(deletion_choices.keys()),
            key="records_to_delete",
        )
        st.warning("選択したデータは統合CSVからまとめて削除されます。この操作は元に戻せません。")
        if st.button(
            "🗑️ 選択したデータをまとめて削除",
            type="secondary",
            disabled=not selected_delete_labels,
        ):
            delete_indices = [deletion_choices[label] for label in selected_delete_labels]
            if delete_results(delete_indices):
                set_flash(f"{len(delete_indices)}件の検出結果を削除しました。過去の場所は保持しています。")
                st.session_state.pop("records_to_delete", None)
                st.rerun()
            else:
                st.error("データを削除できませんでした。")

    st.caption(f"統合CSVの保存先: {CSV_PATH}")
    st.caption("上のダウンロードボタンから、CSVとHTMLをセットで取得できます。")
    st.stop()


# -------------------------------------------------
# 6. 画像アップロード
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "パイプの画像を選択してください",
    type=["jpg", "jpeg", "png"],
    key="pipe_image_uploader",
)

if uploaded_file:
    upload_bytes = uploaded_file.getvalue()
    upload_id = hashlib.sha256(upload_bytes).hexdigest()

    if st.session_state.current_upload_id != upload_id:
        reset_current_image()
        st.session_state.current_upload_id = upload_id

    original_image = Image.open(io.BytesIO(upload_bytes)).convert("RGB")
    base_image = resize_with_aspect_ratio(original_image, TARGET_SIZE)
    display_width, display_height = base_image.size

    # -------------------------------------------------
    # PHASE: UPLOAD
    # -------------------------------------------------
    if st.session_state.phase == "upload":
        st.image(base_image, caption="アップロード画像", width="stretch")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✂️ 検出エリアを編集（除外設定）", use_container_width=True):
                st.session_state.phase = "edit"
                set_flash("除外エリアの編集画面に切り替えました。")
                st.rerun()
        with col2:
            if st.button("🚀 パイプを検出開始", type="primary", use_container_width=True):
                st.session_state.phase = "detect"
                set_flash("検出画面に切り替えました。")
                st.rerun()

    # -------------------------------------------------
    # PHASE: EDIT
    # -------------------------------------------------
    elif st.session_state.phase == "edit":
        st.subheader("除外エリアの設定（2点クリックで四角形を作成）")
        edit_img = apply_masks(base_image.copy(), st.session_state.mask_rects)

        if st.session_state.temp_point:
            draw = ImageDraw.Draw(edit_img)
            tx, ty = st.session_state.temp_point
            draw.ellipse([tx - 8, ty - 8, tx + 8, ty + 8], fill="red")

        coords = streamlit_image_coordinates(
            edit_img, key="mask_editor", width=display_width
        )
        if coords:
            curr_c = (coords["x"], coords["y"])
            if curr_c != st.session_state.last_processed_click:
                st.session_state.last_processed_click = curr_c
                if st.session_state.temp_point is None:
                    st.session_state.temp_point = curr_c
                else:
                    p1 = st.session_state.temp_point
                    p2 = curr_c
                    rect = (
                        min(p1[0], p2[0]),
                        min(p1[1], p2[1]),
                        max(p1[0], p2[0]),
                        max(p1[1], p2[1]),
                    )
                    st.session_state.mask_rects.append(rect)
                    st.session_state.temp_point = None
                st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ 確定して検出へ", type="primary"):
                st.session_state.phase = "detect"
                set_flash("検出画面に切り替えました。")
                st.rerun()
        with col2:
            if st.button("↩️ 1つ戻す"):
                if st.session_state.mask_rects:
                    st.session_state.mask_rects.pop()
                    set_flash("最後に追加した除外エリアを1つ戻しました。")
                else:
                    set_flash("戻せる除外エリアはありません。", "warning")
                st.rerun()
        with col3:
            if st.button("❌ キャンセル"):
                st.session_state.phase = "upload"
                set_flash("除外エリアの編集をキャンセルしました。")
                st.rerun()

    # -------------------------------------------------
    # PHASE: DETECT
    # -------------------------------------------------
    elif st.session_state.phase == "detect":
        st.subheader("検出結果の確認と補正")

        if st.session_state.auto_boxes is None:
            inference_img = apply_masks(base_image.copy(), st.session_state.mask_rects)
            with st.spinner("AIがパイプを検出中..."):
                results = model.predict(
                    inference_img, conf=0.5, max_det=2000, verbose=False
                )
                # 検出数・信頼度の詳細表示は画面に出さず、検出位置だけを保持する

                boxes = []
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    boxes.append(xyxy)
                st.session_state.auto_boxes = boxes

        pointer_size = st.slider(
            "検出ポイントのサイズ",
            min_value=4,
            max_value=30,
            value=13,
            step=1,
            help="画像上に表示する検出ポイントの大きさを変更します。",
        )
        erase_threshold = max(12, pointer_size * 2)

        display_np = np.array(base_image.copy())
        for rect in st.session_state.mask_rects:
            cv2.rectangle(
                display_np,
                (int(rect[0]), int(rect[1])),
                (int(rect[2]), int(rect[3])),
                (40, 40, 40),
                -1,
            )

        for x1, y1, x2, y2 in st.session_state.auto_boxes:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(display_np, (cx, cy), pointer_size, (255, 255, 0), -1)
            cv2.circle(display_np, (cx, cy), pointer_size + 1, (0, 0, 0), 1)

        for px, py in st.session_state.points:
            cv2.circle(display_np, (int(px), int(py)), pointer_size, (255, 0, 0), -1)
            cv2.circle(display_np, (int(px), int(py)), pointer_size + 1, (255, 255, 255), 1)

        total_pipes = len(st.session_state.auto_boxes) + len(st.session_state.points)
        st.success(
            f"合計：**{total_pipes}** 本 "
            f"(AI自動: {len(st.session_state.auto_boxes)} / "
            f"手動追加: {len(st.session_state.points)})"
        )

        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            mode = st.radio(
                "クリック操作を選択:",
                ["📍 ポイント追加", "🧽 消しゴム（クリックで削除）"],
            )
            if st.button("🔄 全リセット（手動追加分のみ）"):
                st.session_state.points = []
                set_flash("手動追加したポイントをすべてリセットしました。")
                st.rerun()

        fix_coords = streamlit_image_coordinates(
            Image.fromarray(display_np), key="manual_fix", width=display_width
        )
        if fix_coords:
            curr_p = (fix_coords["x"], fix_coords["y"])
            if curr_p != st.session_state.last_processed_click:
                st.session_state.last_processed_click = curr_p
                if "追加" in mode:
                    st.session_state.points.append(curr_p)
                else:
                    hit_detected = False
                    new_auto = []
                    for box in st.session_state.auto_boxes:
                        bx, by = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        dist = ((curr_p[0] - bx) ** 2 + (curr_p[1] - by) ** 2) ** 0.5
                        if dist < erase_threshold and not hit_detected:
                            hit_detected = True
                            continue
                        new_auto.append(box)
                    st.session_state.auto_boxes = new_auto
                    if not hit_detected:
                        st.session_state.points = [
                            point for point in st.session_state.points
                            if (
                                (curr_p[0] - point[0]) ** 2
                                + (curr_p[1] - point[1]) ** 2
                            ) ** 0.5 > erase_threshold
                        ]
                st.rerun()

        # -------------------------------------------------
        # 保存情報
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("保存情報")

        # CSVから読み込んだ場所も、通常の保存済み場所も選択肢に表示
        locations = load_locations()
        col1, col2 = st.columns(2)
        with col1:
            group_name = st.text_input("グループ名（自由に変更できます）", value="A")
        with col2:
            location_choice = st.selectbox(
                "過去の場所から選択（任意）",
                ["新しい場所を入力"] + locations,
            )

        if location_choice == "新しい場所を入力":
            location_name = st.text_input(
                "場所", placeholder="例：倉庫1、屋外置き場"
            )
        else:
            location_name = location_choice
            st.text_input("選択した場所", value=location_name, disabled=True)

        user_comment = st.text_area("備考")

        if st.button("💾 判定結果を保存", type="primary", use_container_width=True):
            if not group_name.strip():
                st.error("グループ名を入力してください。")
            elif not location_name.strip():
                st.error("場所を入力するか、過去の場所を選択してください。")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                record = {
                    "日時": timestamp,
                    "グループ": group_name.strip(),
                    "場所": location_name.strip(),
                    "AI検出数": len(st.session_state.auto_boxes),
                    "手動追加数": len(st.session_state.points),
                    "最終合計": total_pipes,
                    "備考": user_comment.strip(),
                }

                # 画像ファイルはローカルのdata/images等には保存しない
                append_result(record)
                add_location(location_name.strip())

                # 画像ファイル単体は保存せず、HTMLファイル内に画像を埋め込む
                image_buffer = io.BytesIO()
                Image.fromarray(display_np).save(image_buffer, format="PNG")
                image_bytes = image_buffer.getvalue()
                html_text = make_html_report(record, image_bytes)

                reports_dir = DATA_DIR / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                export_name = export_basename(record)
                (reports_dir / f"{export_name}.html").write_text(
                    html_text, encoding="utf-8"
                )

                st.session_state.saved_zip_data = create_saved_record_export(record)
                st.session_state.saved_filename = f"{export_name}_CSV_HTML.zip"
                st.session_state.zip_downloaded = False
                st.session_state.saved_message = (
                    "判定結果を保存しました。ZIPには保存済みの全検出結果を含めます。"
                    "ダウンロード後は検出結果データを消去し、過去の場所は残します。"
                )
                st.rerun()

        if st.session_state.saved_zip_data and not st.session_state.zip_downloaded:
            st.success(st.session_state.saved_message or "保存しました。")
            st.download_button(
                "⬇️ CSV＋検出結果HTMLをダウンロード（ZIP）",
                data=st.session_state.saved_zip_data,
                file_name=st.session_state.saved_filename,
                mime="application/zip",
                type="primary",
                on_click="ignore",
            )
            st.caption("ダウンロード後、必要であれば下のボタンからアプリ内の検出結果を削除できます。")
            if st.button("🗑️ ダウンロード後に検出結果をすべて削除", key="clear_after_detection_download"):
                clear_detection_data_after_download()
                st.rerun()

        if st.button("⏮️ 次の画像へ"):
            reset_current_image()
            set_flash("次の画像を選択できる状態に戻しました。")
            st.rerun()

else:
    st.info("パイプの画像を1枚選択してください。")
