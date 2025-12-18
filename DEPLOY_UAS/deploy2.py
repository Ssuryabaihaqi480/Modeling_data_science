import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# LOAD DATA (AUTO-DETECT COLUMN)
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "Income Inequality in South Africa_Dataset.csv",
        sep=";",
        encoding="latin1",
        engine="python"
    )

    # normalisasi nama kolom
    df.columns = df.columns.str.strip().str.lower()

    # deteksi kolom tahun
    year_col = None
    for c in df.columns:
        if "year" in c or "tahun" in c or "time" in c:
            year_col = c
            break

    # deteksi kolom gini
    gini_col = None
    for c in df.columns:
        if "gini" in c:
            gini_col = c
            break

    if year_col is None or gini_col is None:
        st.error("Kolom tahun atau gini tidak ditemukan di CSV")
        st.stop()

    # rename agar konsisten
    df = df.rename(columns={
        year_col: "year",
        gini_col: "gini_disp"
    })

    # pastikan numerik
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gini_disp"] = pd.to_numeric(df["gini_disp"], errors="coerce")

    # bersihkan data
    df = df.dropna(subset=["year", "gini_disp"])
    df = df.sort_values("year")

    return df


# ===============================
# MODELING PAGE (DES)
# ===============================
def modeling_page(df):
    st.title("📊 Modeling: Double Exponential Smoothing (DES)")

    alpha = st.slider("Nilai Alpha (α)", 0.1, 0.9, 0.8)

    if st.button("🔍 Jalankan Modeling"):
        tahun = df["year"].values
        y = df["gini_disp"].values

        # DES Brown
        S1 = [y[0]]
        S2 = [y[0]]

        for t in range(1, len(y)):
            S1.append(alpha * y[t] + (1 - alpha) * S1[t-1])
            S2.append(alpha * S1[t] + (1 - alpha) * S2[t-1])

        a = [2*S1[i] - S2[i] for i in range(len(y))]
        b = [(alpha/(1-alpha)) * (S1[i] - S2[i]) for i in range(len(y))]

        forecast = [np.nan]
        for i in range(1, len(y)):
            forecast.append(a[i-1] + b[i-1])

        # evaluasi
        actual = y[1:]
        pred = np.array(forecast[1:])

        mae = np.mean(np.abs(actual - pred))
        mse = np.mean((actual - pred)**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100

        # forecast 5 tahun
        future_years = np.arange(tahun[-1] + 1, tahun[-1] + 6)
        future_forecast = [a[-1] + b[-1]*m for m in range(1, 6)]

        df_future = pd.DataFrame({
            "Year": future_years,
            "Forecast_GINI_Disp": future_forecast
        })

        # OUTPUT
        st.subheader("📈 Evaluasi Model")
        st.write(f"MAE  : {mae:.4f}")
        st.write(f"MSE  : {mse:.4f}")
        st.write(f"RMSE : {rmse:.4f}")
        st.write(f"MAPE : {mape:.2f}%")

        st.subheader("🔮 Forecast 5 Tahun ke Depan")
        st.dataframe(df_future)

        # plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tahun, y, marker="o", label="Actual")
        ax.plot(
            list(tahun) + list(future_years),
            list(forecast) + list(future_forecast),
            linestyle="--",
            marker="x",
            label="Forecast DES"
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Gini Dispersion")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
        st.success("Modeling berhasil dijalankan ✅")


# ===============================
# MAIN
# ===============================
def main():
    st.sidebar.title("Menu")
    st.sidebar.info("Tahap Modeling (DES)")
    df = load_data()
    modeling_page(df)


if __name__ == "__main__":
    main()
