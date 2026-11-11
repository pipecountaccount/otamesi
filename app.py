import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np
import cv2
import io
import zipfile
from datetime import datetime
from pathlib import Path
from streamlit_image_coordinates import streamlit_image_coordinates

from data_manager import (
    add_location,
    append_result,
    load_locations,
    load_results,
    safe_filename_part,
    IMAGES_DIR,
    CSV_PATH,
)

# -------------------------------------------------
# 1. 初期設定とモデルロード
# -------------------------------------------------

st.set_page_config(
    page_title="PINEAPPLE Pipe Counter",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = "last_openvino_model"
    return YOLO(model_path)

model = load_model()

TARGET_SIZE = 960


# -------------------------------------------------
# 2. Session State の初期化
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

    for r in rects:
        draw.rectangle(
            [r[0], r[1], r[2], r[3]],
            fill="black"
        )

    return img


def resize_with_aspect_ratio(img, target_max_side):
    """アスペクト比を維持してリサイズする"""

    w, h = img.size

    if w > h:
        new_w = target_max_side
        new_h = int(h * (target_max_side / w))
    else:
        new_h = target_max_side
        new_w = int(w * (target_max_side / h))

    return img.resize(
        (new_w, new_h),
        Image.LANCZOS
    )


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


# -------------------------------------------------
# 4. 画面切り替え
# -------------------------------------------------

st.title("PINEAPPLE 🍍")

page = st.sidebar.radio(
    "画面",
    ["画像を判定する", "保存データを確認する"]
)


# -------------------------------------------------
# 5. 保存データ確認画面
# -------------------------------------------------

if page == "保存データを確認する":

    st.header("保存データの確認")

    records = load_results()

    if not records:
        st.info(
            "まだ保存データがありません。画像を判定して保存してください。"
        )

    else:
        records = list(reversed(records))

        labels = [
            f'{r.get("日時", "")} | '
            f'{r.get("グループ", "")} | '
            f'{r.get("場所", "")} | '
            f'{r.get("最終合計", "")}本'
            for r in records
        ]

        selected = st.selectbox(
            "確認するデータを選択",
            range(len(records)),
            format_func=lambda i: labels[i]
        )

        record = records[selected]

        st.subheader("CSVの内容")

        st.dataframe(
            [record],
            use_container_width=True
        )

        image_name = record.get("画像ファイル名", "")
        image_path = IMAGES_DIR / image_name

        if image_name and image_path.exists():

            st.subheader("対応する画像")

            st.image(
                str(image_path),
                caption=image_name,
                use_container_width=True
            )

        else:
            st.warning(
                "対応する画像ファイルが見つかりません。"
            )

        st.caption(
            f"統合CSVの保存先: {CSV_PATH}"
        )

        if CSV_PATH.exists():

            st.download_button(
                "統合CSVをダウンロード",
                data=CSV_PATH.read_bytes(),
                file_name="combined.csv",
                mime="text/csv",
            )

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

    upload_id = (
        f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
    )

    if st.session_state.current_upload_id != upload_id:

        reset_current_image()

        st.session_state.current_upload_id = upload_id

    original_image = Image.open(
        io.BytesIO(uploaded_file.getvalue())
    ).convert("RGB")

    base_image = resize_with_aspect_ratio(
        original_image,
        TARGET_SIZE
    )

    display_width, display_height = base_image.size


    # -------------------------------------------------
    # PHASE: UPLOAD
    # -------------------------------------------------

    if st.session_state.phase == "upload":

        st.image(
            base_image,
            caption="アップロード画像",
            width=display_width
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "✂️ 検出エリアを編集（除外設定）",
                use_container_width=True
            ):

                st.session_state.phase = "edit"
                st.rerun()

        with col2:

            if st.button(
                "🚀 パイプを検出開始",
                type="primary",
                use_container_width=True
            ):

                st.session_state.phase = "detect"
                st.rerun()


    # -------------------------------------------------
    # PHASE: EDIT
    # -------------------------------------------------

    elif st.session_state.phase == "edit":

        st.subheader(
            "除外エリアの設定 (2点クリックで四角形を作成)"
        )

        edit_img = apply_masks(
            base_image.copy(),
            st.session_state.mask_rects
        )

        if st.session_state.temp_point:

            draw = ImageDraw.Draw(edit_img)

            tx, ty = st.session_state.temp_point

            draw.ellipse(
                [tx-8, ty-8, tx+8, ty+8],
                fill="red"
            )

        coords = streamlit_image_coordinates(
            edit_img,
            key="mask_editor",
            width=display_width
        )

        if coords:

            curr_c = (
                coords["x"],
                coords["y"]
            )

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
                        max(p1[1], p2[1])
                    )

                    st.session_state.mask_rects.append(rect)

                    st.session_state.temp_point = None

                st.rerun()

        col1, col2, col3 = st.columns(3)

        with col1:

            if st.button(
                "✅ 確定して検出へ",
                type="primary"
            ):

                st.session_state.phase = "detect"
                st.rerun()

        with col2:

            if st.button("↩️ 1つ戻す"):

                if st.session_state.mask_rects:
                    st.session_state.mask_rects.pop()

                st.rerun()

        with col3:

            if st.button("❌ キャンセル"):

                st.session_state.phase = "upload"
                st.rerun()


    # -------------------------------------------------
    # PHASE: DETECT
    # -------------------------------------------------

    elif st.session_state.phase == "detect":

        st.subheader("検出結果の確認と補正")

        # ---------------------------------------------
        # AIによるパイプ検出
        # ---------------------------------------------

        if st.session_state.auto_boxes is None:

            inference_img = apply_masks(
                base_image.copy(),
                st.session_state.mask_rects
            )

            with st.spinner("AIがパイプを検出中..."):

                results = model.predict(
                    inference_img,
                    conf=0.5,
                    max_det=2000,
                    verbose=False
                )

                # AIの検出数
                detection_count = len(results[0].boxes)

                st.write(
                    "モデルが検出した数:",
                    detection_count
                )

                # 信頼度の確認
                if detection_count > 0:

                    st.write(
                        "信頼度の最低値:",
                        float(results[0].boxes.conf.min())
                    )

                    st.write(
                        "信頼度の最高値:",
                        float(results[0].boxes.conf.max())
                    )

                # 検出位置を保存
                boxes = []

                for box in results[0].boxes:

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    boxes.append([
                        x1,
                        y1,
                        x2,
                        y2
                    ])

                st.session_state.auto_boxes = boxes

                st.write(
                    "AIの検出数（確認用）:",
                    len(st.session_state.auto_boxes)
                )

        # ---------------------------------------------
        # ポインタサイズの調整
        # ---------------------------------------------

        pointer_size = st.slider(
            "検出ポイントのサイズ",
            min_value=4,
            max_value=30,
            value=13,
            step=1,
            help="画像上に表示する検出ポイントの大きさを変更します。",
        )

        erase_threshold = max(
            12,
            pointer_size * 2
        )


        # ---------------------------------------------
        # 検出結果の画像表示
        # ---------------------------------------------

        display_np = np.array(
            base_image.copy()
        )

        # 除外エリアの表示
        for r in st.session_state.mask_rects:

            cv2.rectangle(
                display_np,
                (int(r[0]), int(r[1])),
                (int(r[2]), int(r[3])),
                (40, 40, 40),
                -1
            )

        # AI検出ポイント
        for (x1, y1, x2, y2) in st.session_state.auto_boxes:

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.circle(
                display_np,
                (cx, cy),
                pointer_size,
                (255, 255, 0),
                -1
            )

            cv2.circle(
                display_np,
                (cx, cy),
                pointer_size + 1,
                (0, 0, 0),
                1
            )

        # 手動追加ポイント
        for (px, py) in st.session_state.points:

            cv2.circle(
                display_np,
                (int(px), int(py)),
                pointer_size,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                display_np,
                (int(px), int(py)),
                pointer_size + 1,
                (255, 255, 255),
                1
            )


        # ---------------------------------------------
        # 合計本数
        # ---------------------------------------------

        total_pipes = (
            len(st.session_state.auto_boxes)
            + len(st.session_state.points)
        )

        st.success(
            f"合計： **{total_pipes}** 本 "
            f"(AI自動: {len(st.session_state.auto_boxes)} / "
            f"手動追加: {len(st.session_state.points)})"
        )


        # ---------------------------------------------
        # 追加・削除操作
        # ---------------------------------------------

        col_ctrl1, col_ctrl2 = st.columns([1, 2])

        with col_ctrl1:

            mode = st.radio(
                "クリック操作を選択:",
                [
                    "📍 ポイント追加",
                    "🧽 消しゴム (クリックで削除)"
                ]
            )

            if st.button("🔄 全リセット (手動分のみ)"):

                st.session_state.points = []

                st.rerun()


        # ---------------------------------------------
        # 画像クリック処理
        # ---------------------------------------------

        fix_coords = streamlit_image_coordinates(
            Image.fromarray(display_np),
            key="manual_fix",
            width=display_width
        )

        if fix_coords:

            curr_p = (
                fix_coords["x"],
                fix_coords["y"]
            )

            if curr_p != st.session_state.last_processed_click:

                st.session_state.last_processed_click = curr_p

                if "追加" in mode:

                    st.session_state.points.append(curr_p)

                else:

                    hit_detected = False
                    new_auto = []

                    for box in st.session_state.auto_boxes:

                        bx = (box[0] + box[2]) / 2
                        by = (box[1] + box[3]) / 2

                        dist = (
                            (curr_p[0] - bx) ** 2
                            + (curr_p[1] - by) ** 2
                        ) ** 0.5

                        if dist < erase_threshold and not hit_detected:

                            hit_detected = True
                            continue

                        new_auto.append(box)

                    st.session_state.auto_boxes = new_auto

                    if not hit_detected:

                        st.session_state.points = [
                            p for p in st.session_state.points
                            if (
                                (curr_p[0] - p[0]) ** 2
                                + (curr_p[1] - p[1]) ** 2
                            ) ** 0.5 > erase_threshold
                        ]

                st.rerun()


        # ---------------------------------------------
        # 保存情報の入力
        # ---------------------------------------------

        st.markdown("---")

        st.subheader("保存情報")

        locations = load_locations()

        col1, col2 = st.columns(2)

        with col1:

            group_name = st.text_input(
                "グループ名（自由に変更できます）",
                value="A"
            )

        with col2:

            location_choice = st.selectbox(
                "過去の場所から選択（任意）",
                ["新しい場所を入力"] + locations
            )

        if location_choice == "新しい場所を入力":

            location_name = st.text_input(
                "場所",
                placeholder="例：倉庫1、屋外置き場"
            )

        else:

            location_name = location_choice

            st.text_input(
                "選択した場所",
                value=location_name,
                disabled=True
            )

        user_comment = st.text_area("備考")

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        safe_group = safe_filename_part(
            group_name,
            "group"
        )

        safe_location = safe_filename_part(
            location_name,
            "no_location"
        )

        image_filename = (
            f"{safe_group}_{safe_location}_{timestamp}.png"
        )


        # ---------------------------------------------
        # 保存処理
        # ---------------------------------------------

        if st.button(
            "💾 判定結果を保存",
            type="primary",
            use_container_width=True
        ):

            if not group_name.strip():

                st.error("グループ名を入力してください。")

            elif not location_name.strip():

                st.error(
                    "場所を入力するか、過去の場所を選択してください。"
                )

            else:

                # 画像を保存
                image_path = IMAGES_DIR / image_filename

                Image.fromarray(display_np).save(
                    image_path,
                    format="PNG"
                )

                # CSVに保存する情報
                record = {
                    "日時": timestamp,
                    "グループ": group_name.strip(),
                    "場所": location_name.strip(),
                    "AI検出数": len(st.session_state.auto_boxes),
                    "手動追加数": len(st.session_state.points),
                    "最終合計": total_pipes,
                    "備考": user_comment.strip(),
                    "画像ファイル名": image_filename,
                }

                append_result(record)

                add_location(
                    location_name.strip()
                )

                # ZIPファイルを作成
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(
                    zip_buffer,
                    "w",
                    zipfile.ZIP_DEFLATED
                ) as z:

                    z.write(
                        image_path,
                        arcname=f"images/{image_filename}"
                    )

                    if CSV_PATH.exists():

                        z.writestr(
                            "combined.csv",
                            CSV_PATH.read_bytes()
                        )

                st.session_state.saved_zip_data = (
                    zip_buffer.getvalue()
                )

                st.session_state.saved_filename = (
                    f"pipe_report_{safe_group}_{safe_location}_{timestamp}.zip"
                )

                st.session_state.saved_message = (
                    f"{image_filename} と統合CSVに保存しました。"
                )


        # ---------------------------------------------
        # 保存完了・ダウンロード
        # ---------------------------------------------

        if st.session_state.saved_zip_data:

            st.success(
                st.session_state.saved_message or "保存しました。"
            )

            st.download_button(
                "⬇️ 今回の画像＋統合CSVをダウンロード（ZIP）",
                data=st.session_state.saved_zip_data,
                file_name=st.session_state.saved_filename,
                mime="application/zip",
                type="primary",
            )


        # ---------------------------------------------
        # 次の画像へ
        # ---------------------------------------------

        if st.button("⏮️ 次の画像へ"):

            reset_current_image()

            st.rerun()


else:

    st.info(
        "パイプの画像を1枚選択してください。"
    )