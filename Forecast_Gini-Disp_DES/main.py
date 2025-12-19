import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ====================== PAGE CONFIG & STYLE ======================
st.set_page_config(page_title="Income Inequality Forecast - CRISP-DM", layout="wide")

st.markdown("""
<style>
    .main {background-color: #0E1117; color: #E5E7EB;}
    .stApp {background-color: #0E1117;}
    h1, h2, h3, h4, h5, h6 {color: #00E396; font-weight: bold;}
    .stTextArea label, .stNumberInput label, .stSelectbox label, .stSlider label {color: #E5E7EB !important;}
    
    .metric-card {
        background: linear-gradient(135deg, #1e242f, #2a3244);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.6);
        text-align: center;
        border: 1px solid #334155;
    }
    
    .info-card {
        background: #1a202c;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #00E396;
        height: 100%;
    }
    
    .process-header {
        background: linear-gradient(135deg, #1e242f, #2a3244);
        padding: 25px;
        border-radius: 16px;
        border-left: 5px solid #00E396;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    
    .process-step {
        background: #1a202c;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin: 10px 0;
    }
    
    .highlight-box {
        background: rgba(0, 227, 150, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00E396;
        margin: 10px 0;
    }
    
    .step-number {
        background: #00E396;
        color: #0E1117;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .stButton > button {
        background: #00E396 !important;
        color: black !important;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: #00ffb8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Income Inequality in South Africa_Dataset.xlsx")
    df = pd.read_excel(file_path)
    df = df.sort_values(by='Year')
    return df

df_raw = load_data()

# ====================== SIDEBAR NAVIGATION ======================
with st.sidebar:
    st.markdown("## 🧭 Navigasi CRISP-DM")
    st.markdown("---")
    
    menu = st.radio(
        "Pilih Proses:",
        [
            "💼 Business Understanding",
            "🔍 Data Understanding", 
            "🧹 Data Preparation",
            "🤖 Modeling",
            "✅ Evaluation"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='padding: 15px; background: #1a202c; border-radius: 10px; border-left: 3px solid #00E396;'>
        <small style='color: #E5E7EB;'>
            <strong>CRISP-DM</strong><br>
            Cross-Industry Standard Process for Data Mining
        </small>
    </div>
    """, unsafe_allow_html=True)

# ====================== PAGE CONTENT ======================

# ==================== 1. BUSINESS UNDERSTANDING ====================
if menu == "💼 Business Understanding":
    st.markdown("# 💼 Business Understanding")
    st.markdown("*Memahami konteks bisnis dan tujuan proyek*")
    st.markdown("---")
    
    # Header Card
    st.markdown("""
    <div class='process-header'>
        <h3>🎯 Latar Belakang Masalah</h3>
        <p>Afrika Selatan merupakan salah satu negara dengan tingkat ketimpangan pendapatan tertinggi di dunia. 
        Koefisien Gini adalah indikator utama yang digunakan untuk mengukur distribusi pendapatan dalam suatu populasi.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>📊 Tujuan Bisnis</h4>
            <ul>
                <li>Memahami tren ketimpangan pendapatan di Afrika Selatan</li>
                <li>Memprediksi nilai Gini Coefficient di masa depan</li>
                <li>Memberikan insight untuk kebijakan ekonomi</li>
                <li>Mendukung pengambilan keputusan berbasis data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>🎯 Tujuan Data Mining</h4>
            <ul>
                <li>Membangun model forecasting time series</li>
                <li>Menggunakan Double Exponential Smoothing (Holt's Method)</li>
                <li>Mengoptimalkan parameter α untuk akurasi terbaik</li>
                <li>Mencapai MAPE < 10% (akurasi baik)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h2>🌍</h2>
            <h3>Afrika Selatan</h3>
            <p>Negara dengan ketimpangan tertinggi</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h2>📈</h2>
            <h3>{len(df_raw)} Data Points</h3>
            <p>Historical data tersedia</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h2>🔮</h2>
            <h3>Forecasting</h3>
            <p>Prediksi nilai Gini masa depan</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='highlight-box'>
        <strong>💡 Pertanyaan Bisnis Utama:</strong><br>
        "Bagaimana tren ketimpangan pendapatan di Afrika Selatan dalam beberapa tahun ke depan, 
        dan apakah kebijakan yang ada sudah efektif dalam mengurangi ketimpangan?"
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    # david, business under
    st.subheader("📚 Metodologi CRISP-DM")
    
    with st.expander("📖 Baca Penjelasan Lengkap Siklus CRISP-DM"):
        st.markdown("""
        **1. Business Understanding**
        Tahap Business Understanding merupakan tahap awal dalam proses data science yang bertujuan untuk memahami permasalahan bisnis secara menyeluruh. Pada tahap ini, fokus utama adalah mengidentifikasi tujuan bisnis, permasalahan yang ingin diselesaikan, serta menentukan tujuan analisis data yang selaras dengan kebutuhan bisnis. Selain itu, dilakukan penentuan ruang lingkup proyek, kriteria keberhasilan, serta asumsi dan batasan yang mungkin memengaruhi proses analisis. Hasil dari tahap ini adalah rumusan masalah yang jelas dan terukur sehingga proses data science dapat memberikan solusi yang relevan dan bernilai bagi pengambilan keputusan.

        **2. Data Understanding**
        Tahap Data Understanding bertujuan untuk memahami karakteristik data yang akan digunakan. Proses ini dimulai dengan pengumpulan data awal, dilanjutkan dengan eksplorasi data untuk mengetahui struktur, pola, dan kualitas data. Pada tahap ini juga dilakukan identifikasi terhadap permasalahan data seperti data hilang (missing values), data tidak konsisten, outlier, dan anomali. Pemahaman data yang baik sangat penting agar proses selanjutnya dapat berjalan dengan tepat dan hasil analisis tidak menyesatkan.

        **3. Data Preparation**
        Tahap Data Preparation merupakan proses pengolahan data agar siap digunakan dalam pemodelan. Kegiatan pada tahap ini meliputi pembersihan data, penghapusan atau penanganan data yang hilang, transformasi data, normalisasi, serta pemilihan atribut yang relevan. Tahap ini sering kali memakan waktu paling lama dalam proyek data science karena kualitas model sangat bergantung pada kualitas data yang digunakan. Output dari tahap ini adalah dataset akhir yang telah bersih dan terstruktur dengan baik.

        **4. Modeling**
        Tahap Modeling adalah proses penerapan teknik atau algoritma data science terhadap data yang telah dipersiapkan. Pada tahap ini, dipilih metode pemodelan yang sesuai dengan tujuan analisis, seperti regresi, klasifikasi, clustering, atau peramalan (forecasting). Model kemudian dilatih menggunakan data yang tersedia dan dilakukan penyesuaian parameter agar menghasilkan performa terbaik. Dalam praktiknya, sering dilakukan beberapa percobaan model untuk memperoleh hasil yang paling optimal.

        **5. Evaluation**
        Tahap Evaluation bertujuan untuk menilai kinerja model yang telah dibangun. Evaluasi dilakukan menggunakan metrik tertentu yang sesuai dengan tujuan bisnis, seperti akurasi, error, MSE, MAPE, atau metrik lainnya.Selain evaluasi teknis, pada tahap ini juga dilakukan penilaian apakah hasil model sudah menjawab permasalahan bisnis yang telah dirumuskan pada tahap Business Understanding. Jika hasil belum memuaskan, proses dapat kembali ke tahap sebelumnya untuk dilakukan perbaikan.

        **6. Deployment**
        Tahap Deployment merupakan tahap akhir dalam CRISP-DM, yaitu penerapan model ke dalam lingkungan nyata. Model yang telah dievaluasi dapat diimplementasikan dalam bentuk sistem informasi, aplikasi, dashboard, atau laporan yang dapat digunakan oleh pengguna akhir. Pada tahap ini juga dilakukan monitoring terhadap kinerja model agar tetap relevan seiring berjalannya waktu. Jika terjadi perubahan kondisi bisnis atau data, proses CRISP-DM dapat diulang kembali untuk melakukan penyesuaian.
        """)

# ==================== 2. DATA UNDERSTANDING ====================
elif menu == "🔍 Data Understanding":
    st.markdown("# � Analisis Kualitas Data - Income Inequality South Africa")
    st.markdown("*Eksplorasi dan pemahaman karakteristik data*")
    st.markdown("---")
    
    st.markdown("""
    <div class='process-header'>
        <h3>📂 Sumber Data</h3>
        <p>Dataset: <strong>Income Inequality in South Africa</strong><br>
        File: Income Inequality in South Africa_Dataset.xlsx</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Preview Data
    st.header("1. Preview Data")
    st.dataframe(df_raw.head(5), use_container_width=True, hide_index=True)
    
    # 2. Informasi Data
    st.header("2. Informasi Data")
    st.text(f"Jumlah Baris: {df_raw.shape[0]}")
    st.text(f"Jumlah Kolom: {df_raw.shape[1]}")
    st.text(f"Kolom: {', '.join(df_raw.columns.tolist())}")
    
    # 3. Statistik Deskriptif
    st.header("3. Statistik Deskriptif")
    st.dataframe(df_raw.describe(), use_container_width=True)
    
    # 4. Visualisasi Tren Time Series
    st.header("4. Visualisasi Tren Time Series")
    df_forecast = df_raw[['Year', 'gini_disp']].copy()
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_forecast['Year'], df_forecast['gini_disp'], marker='o', color='#00E396', linewidth=2)
    ax.set_title("Trend Gini Disposable Income (South Africa)", fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel("Year", fontsize=12, color='white')
    ax.set_ylabel("Gini Disposable Income", fontsize=12, color='white')
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # 5. Ringkasan Kualitas Data
    st.header("5. Ringkasan Kualitas Data")
    
    # Fokus pada variabel utama
    df_q = df_raw[['Year', 'gini_disp']].copy()
    
    # Missing value
    missing_value = df_q['gini_disp'].isnull().sum()
    
    # Duplikasi data
    duplikasi = df_q.duplicated().sum()
    
    # Outlier (metode IQR sederhana)
    Q1 = df_q['gini_disp'].quantile(0.25)
    Q3 = df_q['gini_disp'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outlier = df_q[
        (df_q['gini_disp'] < lower) | (df_q['gini_disp'] > upper)
    ].shape[0]
    
    # Tabel ringkasan kualitas data
    # Tampilkan penjelasan metode IQR
    st.subheader("📊 Metode Deteksi Outlier: IQR (Interquartile Range)")
    st.info("""
    **IQR Method:**
    - **Q1 (Kuartil 1)**: Persentil ke-25 dari data
    - **Q3 (Kuartil 3)**: Persentil ke-75 dari data
    - **IQR**: Q3 - Q1 (jarak antara kuartil)
    - **Lower Bound**: Q1 - 1.5 × IQR
    - **Upper Bound**: Q3 + 1.5 × IQR
    - **Outlier**: Data yang berada di luar batas lower atau upper
    """)
    
    # Tampilkan nilai-nilai perhitungan
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Q1 (25%)", f"{Q1:.4f}")
    with col2:
        st.metric("Q3 (75%)", f"{Q3:.4f}")
    with col3:
        st.metric("IQR", f"{IQR:.4f}")
    with col4:
        st.metric("Outlier Count", outlier)
    
    st.text(f"Lower Bound: {lower:.4f}")
    st.text(f"Upper Bound: {upper:.4f}")
    
    st.markdown("---")
    
    quality_summary = pd.DataFrame({
        "Aspek Kualitas Data": ["Missing Value", "Duplikasi Data", "Outlier"],
        "Jumlah": [missing_value, duplikasi, outlier]
    })
    
    st.dataframe(quality_summary, use_container_width=True, hide_index=True)
    
    # Tampilkan kesimpulan
    if missing_value == 0 and duplikasi == 0 and outlier == 0:
        st.success("✅ Data dalam kondisi baik! Tidak ada missing value, duplikasi, atau outlier.")
    else:
        st.warning(f"⚠️ Ditemukan: {missing_value} missing value, {duplikasi} duplikasi, {outlier} outlier")

# ==================== 3. DATA PREPARATION ====================
elif menu == "🧹 Data Preparation":
    # safii, data preparation
    st.markdown("# 🧹 Data Preparation")
    st.markdown("*Data Cleaning, Transformation, dan Exploration untuk Income Inequality South Africa*")
    st.markdown("---")
    
    # Load data original untuk perbandingan
    df_original = df_raw.copy()
    
    # ========== STEP 1: Data Loading & Initial Exploration ==========
    st.markdown("## 📥 STEP 1: Data Loading & Initial Exploration")
    
    st.markdown("""
    <div class='highlight-box'>
        <strong>📌 Penjelasan Step 1:</strong><br>
        ✓ Membaca file Excel yang berisi data Income Inequality South Africa<br>
        ✓ Menggunakan @st.cache_data untuk optimasi performa (data tidak reload setiap kali interaksi)<br>
        ✓ Menyimpan copy original untuk perbandingan sebelum vs sesudah preprocessing
    </div>
    """, unsafe_allow_html=True)
    
    # Display data info metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{df_original.shape[0]}</h3>
            <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{df_original.shape[1]}</h3>
            <p>Total Columns</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{df_original.isnull().sum().sum()}</h3>
            <p>Missing Values</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # all data
    st.subheader("Data Preview (Original)")
    st.dataframe(df_original, use_container_width=True, hide_index=True)
    
    # Data Info dengan Tabs
    st.subheader("Data Information")
    tabs1 = st.tabs(["📊 Data Types", "📈 Summary Statistics", "❓ Missing Values"])
    
    with tabs1[0]:
        st.write("**Column Data Types:**")
        info_df = pd.DataFrame({
            "Column": df_original.columns,
            "Data Type": df_original.dtypes.astype(str),
            # "Non-Null Count": df_original.count(),
            # "Null Count": df_original.isnull().sum()
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    with tabs1[1]:
        st.write("**Summary Statistics (Descriptive):**")
        st.dataframe(df_original.describe(), use_container_width=True)
    
    with tabs1[2]:
        st.write("**Missing Values per Column:**")
        missing_df = pd.DataFrame({
            "Column": df_original.columns,
            "Missing Count": df_original.isnull().sum(),
            "Missing %": (df_original.isnull().sum() / len(df_original) * 100).round(2)
        })
        missing_with_values = missing_df[missing_df["Missing Count"] > 0]
        if len(missing_with_values) > 0:
            st.dataframe(missing_with_values, use_container_width=True, hide_index=True)
        else:
            st.success("✅ Tidak ada missing values!")
    
    # View Full Dataset
    with st.expander("📊 View Full Dataset"):
        st.dataframe(df_original, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ========== STEP 2: Data Sorting & Interpolation ==========
    st.markdown("## 🔧 STEP 2: Data Sorting & Interpolation")
    
    st.markdown("""
    <div class='highlight-box'>
        <strong>📌 Penjelasan Step 2:</strong><br>
        ✓ Mengurutkan data berdasarkan Year (wajib untuk time series interpolation)<br>
        ✓ Mengidentifikasi kolom numerik (menghilangkan Year dari daftar interpolasi)<br>
        ✓ Melakukan Linear Interpolation untuk mengisi missing values dengan nilai yang proporsional antara dua data terdekat<br>
        ✓ Linear Interpolation cocok karena trend data yang smooth dan consistent
    </div>
    """, unsafe_allow_html=True)
    
    # Sort dan Interpolasi
    df_clean = df_original.sort_values(by='Year').reset_index(drop=True)
    
    # Ambil kolom numerik selain Year
    numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
    if 'Year' in numeric_cols:
        numeric_cols.remove('Year')
    
    # Lakukan interpolasi
    for col in numeric_cols:
        df_clean[col] = df_clean[col].interpolate(method='linear')
    
    st.success("✅ Data telah disort berdasarkan Year dan dilakukan interpolasi linear")
    
    # Tampilkan hasil interpolasi
    st.subheader("Data Setelah Interpolasi")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Missing Values Sebelum Interpolasi:**")
        st.dataframe(df_original.isnull().sum(), use_container_width=True)
    with col2:
        st.write("**Missing Values Sesudah Interpolasi:**")
        st.dataframe(df_clean.isnull().sum(), use_container_width=True)
    
    st.markdown("---")
    
    # ========== STEP 3: Visualisasi Interpolasi ==========
    st.markdown("## 📈 STEP 3: Visualisasi Interpolasi - Before vs After")
    
    st.markdown("""
    <div class='highlight-box'>
        <strong>📌 Penjelasan Step 3:</strong><br>
        ✓ Membandingkan visualisasi data SEBELUM interpolasi (dengan missing values) vs SESUDAH<br>
        ✓ Garis merah (before) menunjukkan data asli dengan gaps pada missing values<br>
        ✓ Garis hijau (after) menunjukkan hasil interpolasi yang smooth dan continuous<br>
        ✓ Membantu kita mengidentifikasi apakah interpolasi dilakukan dengan tepat
    </div>
    """, unsafe_allow_html=True)
    
    # Pilih kolom untuk visualisasi interpolasi
    selected_col_interp = st.selectbox("Pilih Kolom untuk Visualisasi Interpolasi:", numeric_cols)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot sebelum interpolasi (original dengan missing values)
    ax.plot(df_original['Year'], df_original[selected_col_interp],
            'o-', color='#EF4444', label='Before (Raw Data)', alpha=0.7, linewidth=2, markersize=8)
    
    # Plot sesudah interpolasi
    ax.plot(df_clean['Year'], df_clean[selected_col_interp],
            '-', color='#00E396', label='After Interpolation', linewidth=2.5)
    
    ax.set_title(f"Perbandingan Interpolasi: {selected_col_interp}", fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel("Year", fontsize=12, color='white')
    ax.set_ylabel(selected_col_interp, fontsize=12, color='white')
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    ax.legend(facecolor='#1a202c', edgecolor='#334155', labelcolor='white')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # ========== STEP 4: Column Selection & Filtering ==========
    st.markdown("## 🎯 STEP 4: Column Selection & Filtering")
    
    st.markdown("""
    <div class='highlight-box'>
        <strong>📌 Penjelasan Step 4:</strong><br>
        ✓ Memilih kolom yang relevan untuk analisis forecasting dan machine learning<br>
        ✓ Menghilangkan kolom yang tidak diperlukan (noise reduction)<br>
        ✓ Fokus pada variabel yang berkaitan dengan income inequality dan faktor-faktor ekonomi<br>
        ✓ Kolom yang dipilih harus memiliki data berkualitas dan tidak terlalu banyak missing values
    </div>
    """, unsafe_allow_html=True)
    
    # Define selected columns - hanya kolom yang tersedia di dataset
    available_cols = df_clean.columns.tolist()
    selected_cols = ['Year', 'gini_disp']
    
    # Tambahkan kolom lain jika ada
    optional_cols = ['gini_mkt', 'Inflation rate', 'GDP', 'GOVEDU', 'GOVEXP', 'FINDEV 1', 'DEMOCRACY', 'FLABOUR']
    for col in optional_cols:
        if col in available_cols:
            selected_cols.append(col)
    
    # Filter dataframe
    df_filtered = df_clean[selected_cols].copy()
    
    st.subheader("Kolom yang Dipilih untuk Analisis")
    col_descriptions = {
        'Year': 'Tahun pengamatan',
        'gini_disp': 'Gini Coefficient (Disposable Income) - TARGET VARIABLE',
        'gini_mkt': 'Gini Coefficient (Market Income)',
        'Inflation rate': 'Inflation rate (%)',
        'GDP': 'Gross Domestic Product',
        'GOVEDU': 'Government Education Spending',
        'GOVEXP': 'Government Expenditure',
        'FINDEV 1': 'Financial Development Index',
        'DEMOCRACY': 'Democracy Index',
        'FLABOUR': 'Labour Force Participation'
    }
    
    col_info = pd.DataFrame({
        "No": range(1, len(selected_cols) + 1),
        "Kolom": selected_cols,
        "Deskripsi": [col_descriptions.get(col, col) for col in selected_cols]
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)
    
    st.subheader("Data Hasil Filtering")
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)
    
    # Summary Statistics
    # st.subheader("📊 Summary Statistik Filtered Data")
    # st.dataframe(df_filtered.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # ========== STEP 5: Analisis Time Series ==========
    st.markdown("## 📈 STEP 5: Analisis Time Series")
    
    st.markdown("""
    <div class='highlight-box'>
        <strong>📌 Penjelasan Step 5:</strong><br>
        ✓ Menganalisis karakteristik time series dari Gini Coefficient<br>
        ✓ Melakukan uji stationarity menggunakan Augmented Dickey-Fuller (ADF) Test<br>
        ✓ Melakukan differencing untuk membuat data stasioner<br>
        ✓ Dekomposisi time series: Trend, Seasonal, dan Residual<br>
        ✓ Analisis ACF dan PACF untuk identifikasi pola autokorelasi
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare time series data
    df_ts = df_filtered[['Year', 'gini_disp']].set_index('Year')
    
    # 5.1 Visualisasi Time Series
    st.subheader("📊 5.1 Visualisasi Time Series - Gini Coefficient")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_ts.index, df_ts['gini_disp'], marker='o', color='#00E396', linewidth=2, markersize=6)
    ax.fill_between(df_ts.index, df_ts['gini_disp'], alpha=0.2, color='#00E396')
    ax.set_title('Time Series: Gini Dispersion', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Year', fontsize=12, color='white')
    ax.set_ylabel('Gini Dispersion', fontsize=12, color='white')
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # 5.2 Statistik Deskriptif
    st.subheader("📋 5.2 Statistik Deskriptif")
    
    col_ts1, col_ts2 = st.columns(2)
    with col_ts1:
        desc_stats = df_ts['gini_disp'].describe()
        st.dataframe(desc_stats.to_frame().T, use_container_width=True)
    
    with col_ts2:
        st.info("""
        **📊 Interpretasi:**
        - **Mean**: Rata-rata nilai Gini Coefficient
        - **Std**: Standar deviasi (volatilitas)
        - **Min/Max**: Rentang nilai
        - **25%/50%/75%**: Kuartil distribusi
        """)
    
    # 5.3 Uji Stationarity (ADF Test)
    st.subheader("🔬 5.3 Uji Stationarity - Augmented Dickey-Fuller (ADF) Test")
    
    # Original data ADF test
    series_clean = df_ts['gini_disp'].replace([np.inf, -np.inf], np.nan).dropna()
    adf_result = adfuller(series_clean)
    
    col_adf1, col_adf2, col_adf3 = st.columns(3)
    with col_adf1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{adf_result[0]:.4f}</h3>
            <p>ADF Statistic</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_adf2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{adf_result[1]:.4f}</h3>
            <p>p-value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_adf3:
        is_stationary = adf_result[1] < 0.05
        status_text = "✅ Stasioner" if is_stationary else "⚠️ Non-Stasioner"
        status_color = "#00E396" if is_stationary else "#EF4444"
        st.markdown(f"""
        <div class='metric-card' style='border: 2px solid {status_color};'>
            <h3>{status_text}</h3>
            <p>Status (α=0.05)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Critical values
    with st.expander("📊 Lihat Critical Values"):
        cv_df = pd.DataFrame({
            "Significance Level": ["1%", "5%", "10%"],
            "Critical Value": [adf_result[4]['1%'], adf_result[4]['5%'], adf_result[4]['10%']]
        })
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        st.info("""
        **Interpretasi:**
        - Jika **ADF Statistic < Critical Value** → Data stasioner
        - Jika **p-value < 0.05** → Tolak H0, data stasioner
        - Jika **p-value > 0.05** → Terima H0, data non-stasioner
        """)
    
    # 5.4 Differencing
    st.subheader("🔄 5.4 Differencing untuk Stationarity")
    
    tabs_diff = st.tabs(["1st Differencing", "2nd Differencing"])
    
    with tabs_diff[0]:
        # First differencing
        df_diff1 = df_ts['gini_disp'].diff().dropna()
        adf_diff1 = adfuller(df_diff1)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"""
            **ADF Statistic (1st Diff):** {adf_diff1[0]:.4f}  
            **p-value:** {adf_diff1[1]:.4f}  
            **Status:** {'✅ Stasioner' if adf_diff1[1] < 0.05 else '⚠️ Non-Stasioner'}
            """)
        
        with col_d2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_diff1.index, df_diff1, color='#00D1FF', linewidth=2)
            ax.axhline(y=0, color='#EF4444', linestyle='--', alpha=0.5)
            ax.set_title('First Differencing', fontsize=12, fontweight='bold', color='white')
            ax.set_facecolor('#0E1117')
            fig.patch.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    with tabs_diff[1]:
        # Second differencing
        df_diff2 = df_diff1.diff().dropna()
        adf_diff2 = adfuller(df_diff2)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"""
            **ADF Statistic (2nd Diff):** {adf_diff2[0]:.4f}  
            **p-value:** {adf_diff2[1]:.4f}  
            **Status:** {'✅ Stasioner' if adf_diff2[1] < 0.05 else '⚠️ Non-Stasioner'}
            """)
        
        with col_d2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_diff2.index, df_diff2, color='#FEB019', linewidth=2)
            ax.axhline(y=0, color='#EF4444', linestyle='--', alpha=0.5)
            ax.set_title('Second Differencing', fontsize=12, fontweight='bold', color='white')
            ax.set_facecolor('#0E1117')
            fig.patch.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    # 5.5 Dekomposisi Time Series
    st.subheader("🔍 5.5 Dekomposisi Time Series (Trend, Seasonal, Residual)")
    
    try:
        # Dekomposisi dengan period=1 (no seasonality expected)
        decomposition = seasonal_decompose(df_ts['gini_disp'], model='additive', period=1)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Original
        axes[0].plot(df_ts.index, df_ts['gini_disp'], color='#00E396', linewidth=2)
        axes[0].set_ylabel('Original', fontsize=10, color='white')
        axes[0].set_title('Time Series Decomposition', fontsize=14, fontweight='bold', color='white')
        
        # Trend
        axes[1].plot(decomposition.trend.index, decomposition.trend, color='#00D1FF', linewidth=2)
        axes[1].set_ylabel('Trend', fontsize=10, color='white')
        
        # Seasonal
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal, color='#FEB019', linewidth=2)
        axes[2].set_ylabel('Seasonal', fontsize=10, color='white')
        
        # Residual
        axes[3].plot(decomposition.resid.index, decomposition.resid, color='#EF4444', linewidth=2)
        axes[3].set_ylabel('Residual', fontsize=10, color='white')
        axes[3].set_xlabel('Year', fontsize=10, color='white')
        
        for ax in axes:
            ax.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
        
        fig.patch.set_facecolor('#0E1117')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("""
        **📊 Interpretasi Dekomposisi:**
        - **Trend**: Pola jangka panjang dalam data
        - **Seasonal**: Pola berulang dalam periode tertentu
        - **Residual**: Noise atau variasi acak setelah trend dan seasonal dihilangkan
        """)
    except Exception as e:
        st.warning(f"⚠️ Dekomposisi tidak dapat dilakukan: {str(e)}")
    
    # 5.6 ACF dan PACF
    st.subheader("📉 5.6 Autocorrelation Function (ACF) & Partial ACF (PACF)")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ACF plot
    plot_acf(df_diff2, lags=min(20, len(df_diff2)-1), ax=axes[0], color='#00E396')
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold', color='white')
    axes[0].set_facecolor('#0E1117')
    axes[0].tick_params(colors='white')
    
    # PACF plot
    plot_pacf(df_diff2, lags=min(20, len(df_diff2)-1), ax=axes[1], color='#00D1FF')
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold', color='white')
    axes[1].set_facecolor('#0E1117')
    axes[1].tick_params(colors='white')
    
    fig.patch.set_facecolor('#0E1117')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("""
    **📊 Interpretasi ACF & PACF:**
    - **ACF**: Mengukur korelasi antara observasi pada lag berbeda
    - **PACF**: Mengukur korelasi langsung setelah menghilangkan efek lag sebelumnya
    - Digunakan untuk menentukan parameter **p** dan **q** dalam model ARIMA
    """)
    
    st.markdown("---")
    
    # ========== STEP 6: Visualisasi Outlier ==========
    st.markdown("## 📊 STEP 6: Visualisasi Outlier Detection")
    
    st.markdown("""
    <div class='highlight-box'>
        <strong>📌 Penjelasan Step 6:</strong><br>
        ✓ Menggunakan boxplot untuk mendeteksi outlier pada setiap kolom numerik<br>
        ✓ Boxplot menampilkan distribusi data: Q1, Median, Q3, dan nilai ekstrem<br>
        ✓ Outlier ditandai sebagai titik di luar "whiskers" (batas atas/bawah IQR)<br>
        ✓ Membantu mengidentifikasi data yang tidak normal atau anomali
    </div>
    """, unsafe_allow_html=True)
    
    # Pilih kolom numerik kecuali 'Year'
    numeric_cols_outlier = df_filtered.select_dtypes(include='number').columns.tolist()
    if 'Year' in numeric_cols_outlier:
        numeric_cols_outlier.remove('Year')
    
    st.subheader("Boxplot untuk Deteksi Outlier")
    
    # Pilih kolom untuk visualisasi
    selected_col_outlier = st.selectbox(
        "Pilih Kolom untuk Visualisasi Boxplot:", 
        numeric_cols_outlier,
        key="outlier_selectbox"
    )
    
    # Buat boxplot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Data untuk boxplot
    data_to_plot = df_filtered[selected_col_outlier].dropna()
    
    # Boxplot styling
    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.5)
    
    # Styling boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('#00E396')
        patch.set_alpha(0.7)
    
    for whisker in bp['whiskers']:
        whisker.set(color='#00E396', linewidth=2)
    
    for cap in bp['caps']:
        cap.set(color='#00E396', linewidth=2)
    
    for median in bp['medians']:
        median.set(color='#FEB019', linewidth=2)
    
    for flier in bp['fliers']:
        flier.set(marker='o', color='#EF4444', markersize=8, alpha=0.8)
    
    # Styling
    ax.set_title(f"Boxplot - {selected_col_outlier}", fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel(selected_col_outlier, fontsize=12, color='white')
    ax.set_xticks([1])
    ax.set_xticklabels([selected_col_outlier], color='white')
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Statistik Outlier
    st.subheader("📈 Statistik Outlier")
    
    Q1 = data_to_plot.quantile(0.25)
    Q3 = data_to_plot.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data_to_plot[(data_to_plot < lower_bound) | (data_to_plot > upper_bound)]
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{Q1:.4f}</h3>
            <p>Q1 (25%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{Q3:.4f}</h3>
            <p>Q3 (75%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{IQR:.4f}</h3>
            <p>IQR</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat4:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{len(outliers)}</h3>
            <p>Outliers</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detail outlier
    if len(outliers) > 0:
        st.warning(f"⚠️ Ditemukan {len(outliers)} outlier pada kolom {selected_col_outlier}")
        st.markdown(f"**Lower Bound:** {lower_bound:.4f} | **Upper Bound:** {upper_bound:.4f}")
        
        with st.expander("📋 Lihat Detail Outliers"):
            outlier_df = pd.DataFrame({
                "Index": outliers.index,
                "Nilai": outliers.values
            })
            st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    else:
        st.success(f"✅ Tidak ada outlier terdeteksi pada kolom {selected_col_outlier}")
    
    st.markdown("---")
    
    # ========== Summary ==========
    st.markdown("## ✅ Data Preparation Complete!")
    
    st.markdown(f"""
    <div class='highlight-box'>
        <strong>📊 Ringkasan Proses:</strong><br><br>
        ✓ Dari <strong>{df_original.shape[0]}</strong> baris, <strong>{df_original.shape[1]}</strong> kolom awal<br>
        ✓ Setelah filtering: <strong>{df_filtered.shape[0]}</strong> baris, <strong>{df_filtered.shape[1]}</strong> kolom<br>
        ✓ Missing values telah diatasi dengan interpolasi linear<br>
        ✓ Analisis time series: stationarity test, differencing, decomposition, ACF/PACF<br>
        ✓ Outlier telah diidentifikasi dan divisualisasi<br>
        ✓ Data siap untuk Modeling dan Evaluation
    </div>
    """, unsafe_allow_html=True)

# ==================== 4. MODELING ====================
elif menu == "🤖 Modeling":
    st.markdown("# 🤖 Modeling")
    st.markdown("*Pembangunan model Double Exponential Smoothing*")
    st.markdown("---")
    
    st.markdown("""
    <div class='process-header'>
        <h3>📐 Double Exponential Smoothing (Holt's Method)</h3>
        <p>Metode forecasting untuk time series dengan trend linier.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Method Explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>📖 Tentang Metode</h4>
            <p>Double Exponential Smoothing (DES) atau Brown's Method adalah teknik peramalan yang cocok untuk data dengan <strong>trend linier</strong>.</p>
            <p>Metode ini menggunakan dua level smoothing untuk menangkap:</p>
            <ul>
                <li><strong>Level (a)</strong>: Nilai rata-rata yang di-smooth</li>
                <li><strong>Trend (b)</strong>: Arah perubahan data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>🎛️ Parameter Model</h4>
            <p><strong>Alpha (α)</strong>: Smoothing factor (0 < α < 1)</p>
            <ul>
                <li>α mendekati 0 → smoothing lambat, stabil</li>
                <li>α mendekati 1 → responsif, mengikuti data terbaru</li>
            </ul>
            <p><strong>Rekomendasi:</strong> α = 0.1 - 0.3 untuk data stabil</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Formulas
    st.subheader("📝 Formula Double Exponential Smoothing")
    
    st.markdown("""
    <div class='highlight-box'>
        <strong>Single Exponential Smoothing (S'):</strong><br>
        <code>S't = α × Yt + (1 - α) × S't-1</code><br><br>
        <strong>Double Exponential Smoothing (S''):</strong><br>
        <code>S''t = α × S't + (1 - α) × S''t-1</code><br><br>
        <strong>Komponen Level (a):</strong><br>
        <code>at = 2 × S't - S''t</code><br><br>
        <strong>Komponen Trend (b):</strong><br>
        <code>bt = (α / (1 - α)) × (S't - S''t)</code><br><br>
        <strong>Forecast m periode ke depan:</strong><br>
        <code>Ft+m = at + bt × m</code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Demo
    st.subheader("🔬 Demo Perhitungan")
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Parameter Model")
        alpha = st.slider("Alpha (α)", min_value=0.01, max_value=0.99, value=0.60, step=0.01)
        periods_ahead = st.number_input("Periode Prediksi", min_value=1, max_value=20, value=5)
        
        if st.button("🔥 Hitung Forecast", type="primary", use_container_width=True):
            st.session_state.calculate = True
    
    if st.session_state.get("calculate", False):
        # Perhitungan
        # Lakukan interpolasi agar time series tidak bolong
        df_clean = df_raw[['Year', 'gini_disp']].sort_values('Year').reset_index(drop=True)
        df_clean['gini_disp'] = df_clean['gini_disp'].interpolate(method='linear')
        df_clean = df_clean.dropna() # Drop rows yang masih NaN (misal di awal/akhir)

        Y = df_clean['gini_disp'].values.astype(float)
        years = df_clean['Year'].values.astype(int)
        n = len(Y)
        
        S1 = [Y[0]]
        S2 = [Y[0]]
        for t in range(1, n):
            S1.append(alpha * Y[t] + (1 - alpha) * S1[t-1])
            S2.append(alpha * S1[t] + (1 - alpha) * S2[t-1])
        
        a = [2 * S1[i] - S2[i] for i in range(n)]
        b = [((alpha / (1 - alpha)) * (S1[i] - S2[i])) if (1 - alpha) != 0 else 0.0 for i in range(n)]
        
        forecast = [None]
        for i in range(1, n):
            forecast.append(a[i-1] + b[i-1])
        
        # Tabel Hasil
        st.markdown("#### 📋 Tabel Perhitungan")
        table_data = []
        for i in range(n):
            table_data.append({
                "No": i + 1,
                "Tahun": int(years[i]),
                "Gini (Yt)": f"{Y[i]:.4f}",
                "S't": f"{S1[i]:.4f}",
                "S''t": f"{S2[i]:.4f}",
                "at": f"{a[i]:.4f}",
                "bt": f"{b[i]:.4f}",
                "Forecast": f"{forecast[i]:.4f}" if forecast[i] is not None else "-",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
        
        # Prediksi
        # surya, modeling grafik dan forecast periode tertentu
        future_years = [years[-1] + k + 1 for k in range(periods_ahead)]
        future_forecasts = [a[-1] + b[-1] * m for m in range(1, periods_ahead + 1)]
        
        st.markdown(f"#### 🔮 Prediksi {periods_ahead} Tahun ke Depan")
        pred_df = pd.DataFrame({
            "Tahun": future_years,
            "Prediksi Gini": [f"{v:.4f}" for v in future_forecasts]
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # ========== GRAFIK VISUALISASI ==========
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈 Visualisasi: Aktual vs Forecast")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data aktual
        ax.plot(years, Y, marker='o', label='Actual GINI', color='#00E396', linewidth=2, markersize=6)
        
        # Plot forecast in-sample
        forecast_clean = [f if f is not None else np.nan for f in forecast]
        ax.plot(years, forecast_clean, marker='x', linestyle='--', label='Forecast (In-sample)', color='#00D1FF', linewidth=2)
        
        # Plot forecast future
        ax.plot(future_years, future_forecasts, marker='s', linestyle='--', label='Forecast (Future)', color='#FEB019', linewidth=2, markersize=8)
        
        # Garis pemisah
        ax.axvline(x=years[-1], color='#EF4444', linestyle=':', alpha=0.7, label='Cutoff')
        
        # Styling
        ax.set_xlabel('Year', fontsize=12, color='white')
        ax.set_ylabel('GINI Coefficient', fontsize=12, color='white')
        ax.set_title(f'Forecasting Gini Coefficient (α = {alpha})', fontsize=14, fontweight='bold', color='white')
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a202c', edgecolor='#334155', labelcolor='white')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        st.success("✅ Modeling dan visualisasi berhasil dijalankan!")
        
    else:
        st.info("👈 Atur parameter di sidebar dan klik **Hitung Forecast** untuk melihat hasil perhitungan.")

# ==================== 5. EVALUATION ====================
elif menu == "✅ Evaluation":
    st.markdown("# ✅ Evaluation")
    st.markdown("*Evaluasi performa model forecasting*")
    st.markdown("---")
    
    st.markdown("""
    <div class='process-header'>
        <h3>📊 Metrik Evaluasi Model</h3>
        <p>Menggunakan berbagai metrik error untuk mengukur akurasi model forecasting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parameter
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Parameter Evaluasi")
        alpha = st.slider("Alpha (α)", min_value=0.01, max_value=0.99, value=0.60, step=0.01, key="eval_alpha")
        periods_ahead = st.number_input("Periode Prediksi", min_value=1, max_value=20, value=5, key="eval_periods")
        
        if st.button("📊 Evaluasi Model", type="primary", use_container_width=True):
            st.session_state.evaluate = True
    
    # Metrics Explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>📏 Metrik yang Digunakan</h4>
            <ul>
                <li><strong>MAE</strong>: Mean Absolute Error</li>
                <li><strong>MSE</strong>: Mean Squared Error</li>
                <li><strong>RMSE</strong>: Root Mean Squared Error</li>
                <li><strong>MAPE</strong>: Mean Absolute Percentage Error</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>📊 Interpretasi MAPE</h4>
            <table style='width:100%;'>
                <tr><td style='background:#00E396;color:black;padding:5px;border-radius:5px;'>< 5%</td><td style='padding-left:10px;'>Sangat Baik</td></tr>
                <tr><td style='background:#00D1FF;color:black;padding:5px;border-radius:5px;'>5-10%</td><td style='padding-left:10px;'>Baik</td></tr>
                <tr><td style='background:#FEB019;color:black;padding:5px;border-radius:5px;'>10-20%</td><td style='padding-left:10px;'>Cukup</td></tr>
                <tr><td style='background:#EF4444;color:white;padding:5px;border-radius:5px;'>> 20%</td><td style='padding-left:10px;'>Perlu Perbaikan</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.get("evaluate", False):
        # Perhitungan
        # Lakukan interpolasi agar time series tidak bolong
        df_clean = df_raw[['Year', 'gini_disp']].sort_values('Year').reset_index(drop=True)
        df_clean['gini_disp'] = df_clean['gini_disp'].interpolate(method='linear')
        df_clean = df_clean.dropna()

        Y = df_clean['gini_disp'].values.astype(float)
        years = df_clean['Year'].values.astype(int)
        n = len(Y)
        
        S1 = [Y[0]]
        S2 = [Y[0]]
        for t in range(1, n):
            S1.append(alpha * Y[t] + (1 - alpha) * S1[t-1])
            S2.append(alpha * S1[t] + (1 - alpha) * S2[t-1])
        
        a = [2 * S1[i] - S2[i] for i in range(n)]
        b = [((alpha / (1 - alpha)) * (S1[i] - S2[i])) if (1 - alpha) != 0 else 0.0 for i in range(n)]
        
        forecast = [None]
        for i in range(1, n):
            forecast.append(a[i-1] + b[i-1])
        
        # Error calculation
        error = [None]
        abs_error = [None]
        error2 = [None]
        for i in range(1, n):
            f = forecast[i]
            if f is not None:
                e = Y[i] - f
                error.append(e)
                abs_error.append(abs(e))
                error2.append(e**2)
        
        valid_errors = [e for e in error if e is not None]
        valid_indices = [i for i in range(n) if error[i] is not None]
        
        MAE = np.mean(np.abs(valid_errors))
        MSE = np.mean(np.square(valid_errors))
        RMSE = np.sqrt(MSE)
        valid_y = np.array([Y[i] for i in valid_indices if Y[i] != 0])
        valid_abs_error = np.array([abs_error[i] for i in valid_indices if Y[i] != 0])
        MAPE = float(np.nanmean(valid_abs_error / valid_y) * 100)
        
        # Display Metrics
        st.subheader("📊 Hasil Evaluasi Model")
        
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{MAE:.4f}</h3>
                <p>MAE</p>
                <small>Mean Absolute Error</small>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{MSE:.4f}</h3>
                <p>MSE</p>
                <small>Mean Squared Error</small>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{RMSE:.4f}</h3>
                <p>RMSE</p>
                <small>Root Mean Squared Error</small>
            </div>
            """, unsafe_allow_html=True)
        with cols[3]:
            mape_color = "🟢" if MAPE < 5 else "🟡" if MAPE < 10 else "🟠" if MAPE < 20 else "🔴"
            mape_desc = "Sangat Baik" if MAPE < 5 else "Baik" if MAPE < 10 else "Cukup" if MAPE < 20 else "Perlu Perbaikan"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{mape_color} {MAPE:.2f}%</h3>
                <p>MAPE</p>
                <small>{mape_desc}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualization
        st.subheader("📈 Visualisasi: Aktual vs Forecast")
        
        future_years = [years[-1] + k + 1 for k in range(periods_ahead)]
        future_forecasts = [a[-1] + b[-1] * m for m in range(1, periods_ahead + 1)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(years, Y, marker='o', label='Actual GINI', color='#00E396', linewidth=2, markersize=6)
        
        forecast_clean = [f if f is not None else np.nan for f in forecast]
        ax.plot(years, forecast_clean, marker='x', linestyle='--', label='Forecast (In-sample)', color='#00D1FF', linewidth=2)
        ax.plot(future_years, future_forecasts, marker='s', linestyle='--', label='Forecast (Future)', color='#FEB019', linewidth=2, markersize=8)
        
        ax.axvline(x=years[-1], color='#EF4444', linestyle=':', alpha=0.7, label='Cutoff')
        ax.set_xlabel('Year', fontsize=12, color='white')
        ax.set_ylabel('GINI Coefficient', fontsize=12, color='white')
        ax.set_title(f'Forecasting Gini Coefficient (α = {alpha})', fontsize=14, fontweight='bold', color='white')
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a202c', edgecolor='#334155', labelcolor='white')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Conclusion
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='highlight-box'>
            <strong>📝 Kesimpulan</strong><br><br>
            Model Double Exponential Smoothing dengan α = {alpha:.2f} menghasilkan MAPE sebesar <strong>{MAPE:.2f}%</strong> 
            yang termasuk kategori <strong>{mape_desc}</strong>.<br><br>
            Prediksi Gini Coefficient untuk {periods_ahead} tahun ke depan menunjukkan tren 
            {'meningkat' if future_forecasts[-1] > Y[-1] else 'menurun'} dari nilai terakhir {Y[-1]:.4f} 
            menjadi {future_forecasts[-1]:.4f} pada tahun {future_years[-1]}.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Atur parameter di sidebar dan klik **Evaluasi Model** untuk melihat hasil evaluasi.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 20px;'>
    <p>📊 Income Inequality Forecast - CRISP-DM Methodology</p>
    <small>Double Exponential Smoothing (Holt's Method) | Afrika Selatan Dataset</small>
</div>
""", unsafe_allow_html=True)
