import streamlit as st
import pandas as pd
import json
from pathlib import Path

# Настройки страницы
st.set_page_config(page_title="Складской Терминал - Отчеты", layout="wide", page_icon="📦")

st.title("📊 История инвентаризации металлопроката")
st.markdown("---")

OUTPUT_DIR = Path("captures")


# Функция загрузки данных с кэшированием (обновляется раз в 3 секунды при взаимодействии)
@st.cache_data(ttl=3)
def load_scans():
    if not OUTPUT_DIR.exists():
        return pd.DataFrame()

    scans = []
    # Ищем все json файлы с метаданными
    for json_file in OUTPUT_DIR.glob("scan_*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                scans.append(data)
        except Exception:
            continue

    if not scans:
        return pd.DataFrame()

    df = pd.DataFrame(scans)
    # Сортируем: новые сканирования сверху
    df = df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
    return df


df = load_scans()

if df.empty:
    st.info("📭 Пока нет данных. Сделайте первое сканирование на устройстве!")
else:
    # Блок выбора конкретного сканирования
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        options = [f"🕒 {row['timestamp']}  |  📦 {row['count']} шт." for _, row in df.iterrows()]
        selected_idx = st.selectbox("Выберите сканирование для детального просмотра:", range(len(options)),
                                    format_func=lambda x: options[x])

    with col_btn:
        # Кнопка для принудительного обновления (чтобы не ждать ttl)
        if st.button("🔄 Обновить", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    selected_scan = df.iloc[selected_idx]

    # Вывод больших фото
    st.markdown(f"### Результаты сканирования: **{selected_scan['count']} труб**")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Оригинальное фото")
        st.image(str(OUTPUT_DIR / selected_scan['photo']), use_container_width=True)

    with col2:
        st.subheader("🎭 Маска сегментации (AI)")
        st.image(str(OUTPUT_DIR / selected_scan['mask']), use_container_width=True)

    st.markdown("---")

    # Общая таблица истории
    st.subheader("📋 Полный журнал операций")

    # Форматируем таблицу для красивого вывода
    display_df = df[['timestamp', 'count', 'photo', 'mask']].copy()
    display_df.columns = ['⏱️ Время', '📦 Кол-во', '📷 Фото', '🎭 Маска']

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "⏱️ Время": st.column_config.TextColumn(width="medium"),
            "📦 Кол-во": st.column_config.NumberColumn(format="%d шт.", width="small"),
        }
    )