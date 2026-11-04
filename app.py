import streamlit as st


st.title(
    "はじめてのStreamlit"
)


name = st.text_input(
    "名前を入力してください"
)


if name:

    st.success(
        f"こんにちは、{name}さん！"
    )