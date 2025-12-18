import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# LOAD DATA (EXCEL | AUTO-DETECT COLUMN)
# =====================================
@st.cache_data
def load_data():
    df = pd.read_excel(
        "Income Inequality in South Africa_Dataset.xlsx",
        engine="openpyxl"
    )

    # Normalisasi nama kolom
    df.columns = df.columns.str.strip().str.lower()

    # Deteksi kolom tahun
    year_col = None
    for c in df.columns:
        if "year" in c or "tahun" in c or "time" in c:
            year_col = c
            break

    # Deteksi kolom gini
    gini_col = None
    for c in df.columns:
        if "gini" in c:
            gini_col = c
            break

    if year_col is None or gini_col is None:
        st.error("❌ Kolom tahun atau gini tidak ditemukan di file Excel")
        st.stop()

    # Rename agar konsisten
    df = df.rename(columns={
        year_col: "year",
        gini_col: "gini_disp"
    })

    # Konversi numerik
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gini_disp"] = pd.to_numeric(df["gini_disp"], errors="coerce")

    # Bersihkan data
    df = df.dropna(subset=["year", "gini_disp"])
    df = df.sort_values("year").reset_index(drop=True)

    return df


# =====================================
# MODELING PAGE (DES BROWN)
# =====================================
def modeling_page(df):
    st.title("📊 Modeling Double Exponential Smoothing (DES)")
    st.write(
        "Metode **Double Exponential Smoothing (Brown)** digunakan "
        "untuk memodelkan tren ketimpangan pendapatan (Gini Dispersion)."
    )

    alpha = st.slider("Nilai Alpha (α)", 0.1, 0.9, 0.8)

    if st.button("🔍 Jalankan Modeling"):

        tahun = df["year"].values
        y = df["gini_disp"].values

        # =====================================
        # DES BROWN
        # =====================================
        S1 = [y[0]]
        S2 = [y[0]]

        for t in range(1, len(y)):
            S1.append(alpha * y[t] + (1 - alpha) * S1[t - 1])
            S2.append(alpha * S1[t] + (1 - alpha) * S2[t - 1])

        a = [2 * S1[i] - S2[i] for i in range(len(y))]
        b = [(alpha / (1 - alpha)) * (S1[i] - S2[i]) for i in range(len(y))]

        # Forecast in-sample
        forecast = [np.nan]
        for i in range(1, len(y)):
            forecast.append(a[i - 1] + b[i - 1])

        # =====================================
        # EVALUASI MODEL
        # =====================================
        actual = y[1:]
        pred = np.array(forecast[1:])

        mae = np.mean(np.abs(actual - pred))
        mse = np.mean((actual - pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100

        # =====================================
        # FORECAST 5 TAHUN (MULAI 2022)
        # =====================================
        start_year = 2022
        n_forecast = 5

        future_years = np.arange(start_year, start_year + n_forecast)

        future_forecast = [
            a[-1] + b[-1] * m
            for m in range(1, n_forecast + 1)
        ]

        df_future = pd.DataFrame({
            "Year": future_years,
            "Forecast_GINI_Disp": future_forecast
        })

        # =====================================
        # OUTPUT STREAMLIT
        # =====================================
        st.subheader("📈 Evaluasi Model")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("MSE", f"{mse:.4f}")
        col3.metric("RMSE", f"{rmse:.4f}")
        col4.metric("MAPE", f"{mape:.2f}%")

        st.subheader("🔮 Forecast Gini Dispersion (2022–2026)")
        st.dataframe(df_future, use_container_width=True)

        # =====================================
        # VISUALISASI
        # =====================================
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(tahun, y, marker="o", label="Actual Gini Disp")

        plot_years = np.concatenate([tahun, future_years])
        plot_values = np.concatenate([forecast, future_forecast])

        ax.plot(
            plot_years,
            plot_values,
            linestyle="--",
            marker="x",
            label="Forecast DES"
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Gini Dispersion")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
        st.success("✅ Modeling DES berhasil dijalankan")


# =====================================
# MAIN
# =====================================
def main():
    st.sidebar.title("📌 Menu")
    st.sidebar.info("Tahap Modeling (DES Brown)")
    df = load_data()
    modeling_page(df)


if __name__ == "__main__":
    main()
