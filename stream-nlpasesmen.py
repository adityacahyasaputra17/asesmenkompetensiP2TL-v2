import streamlit as st
import pickle
import tempfile
from faster_whisper import WhisperModel
import pandas as pd
import io
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# === Daftar akun asesor (multi-user dengan role) ===
ACCOUNTS = {
    "fathony": {"password": "yamlicha123", "role": "asesor"},
    "winoto": {"password": "winoto123", "role": "asesor"},
    "dedy": {"password": "septawan123", "role": "asesor"},
    "buyung": {"password": "marthinus123", "role": "asesor"},
    "zaenal": {"password": "abidin123", "role": "asesor"},
    "admin": {"password": "admin789", "role": "admin"}
}

# === Inisialisasi session_state ===
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
if "riwayat_asesmen" not in st.session_state:
    st.session_state.riwayat_asesmen = []

# === Fungsi login & logout ===
def login(username, password):
    if username in ACCOUNTS and ACCOUNTS[username]["password"] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = ACCOUNTS[username]["role"]
        st.success(f"✅ Login berhasil, selamat datang {username}")
    else:
        st.error("❌ Username atau password salah")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.info("Anda telah logout.")

# === Halaman Login (Homepage) ===
if not st.session_state.logged_in:
    st.title("🔐 Halaman Login Asesor")
    st.markdown("Masukkan kredensial untuk mengakses sistem asesmen.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        login(username, password)

# === Halaman Asesmen (hanya muncul jika login berhasil) ===
else:
    st.title("🗣️ Uji Asesmen Lisan Pegawai")
    st.markdown(f"Selamat datang, **{st.session_state.username}**.")
    st.markdown(f"Silakan rekam jawaban pegawai, sistem akan mentranskrip dan mengevaluasi kompetensi secara otomatis.")
                
    # Input nama asesi
    nama_asesi = st.text_input("👤 Nama Asesi yang diuji")

    # Input audio
    st.markdown("## 🎙️ Rekam Audio Jawaban")
    audio_file = st.audio_input("Klik tombol ini 🎙️ untuk merekam", label_visibility="visible")

    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
        font-size: 20px;
        height: 70px;
}
    </style>
    """, unsafe_allow_html=True)

    if audio_file is not None and nama_asesi.strip() != "":
        # Simpan audio sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(audio_file.read())
            audio_path = tmpfile.name

        # Transkripsi dengan FasterWhisper
        st.info("⏳ Sedang melakukan transkripsi...")
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, info = whisper_model.transcribe(audio_path, language="id")

        transcript = " ".join([segment.text for segment in segments])

        # Tampilkan transkrip
        st.subheader("📄 Transkrip Jawaban")
        st.write(transcript)

    # Prediksi dengan pipeline
    # Tambahkan tombol untuk memicu prediksi
    if st.button("Lihat Hasil Prediksi"):
        with open("pipeline_asesmen.pkl", "rb") as f:
            pipeline = pickle.load(f)
        prediction = pipeline.predict([transcript])[0]

        st.subheader("📊 Hasil Prediksi")
        if prediction == 1:
            st.success("✅ Pegawai dinilai: **Kompeten**")
        else:
            st.error("⚠️ Pegawai dinilai: **Tidak Kompeten**")

        # Simpan ke riwayat asesmen dengan nama asesi + timestamp
        st.session_state.riwayat_asesmen.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Asesor": st.session_state.username,
            "Asesi": nama_asesi,
            "Transkrip": transcript,
            "Prediksi": "Kompeten" if prediction == 1 else "Tidak Kompeten"
        })

    elif audio_file is not None and nama_asesi.strip() == "":
        st.warning("⚠️ Harap isi nama asesi sebelum memulai asesmen.")

if st.session_state.riwayat_asesmen:
    st.markdown("## ☁️ Analisis Word Cloud")

    # Buat DataFrame riwayat
    hasil_df = pd.DataFrame(st.session_state.riwayat_asesmen)

    # Pilihan mode analisis
    mode = st.radio("Pilih mode Word Cloud:", ["Keseluruhan Peserta", "Per Asesi"])

    if mode == "Keseluruhan Peserta":
        # Gabungkan semua transkrip
        all_text = " ".join(hasil_df["Transkrip"].tolist())
        if all_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    elif mode == "Per Asesi":
        # Pilih nama asesi
        peserta_list = hasil_df["Asesi"].unique().tolist()
        selected_asese = st.selectbox("Pilih Asesi:", peserta_list)

        # Gabungkan transkrip milik asesi terpilih
        text_asese = " ".join(hasil_df[hasil_df["Asesi"] == selected_asese]["Transkrip"].tolist())
        if text_asese.strip():
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_asese)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    st.markdown(f"========================================================================================")
    st.markdown(f"========================================================================================")
    
    # Tampilkan riwayat asesmen
    if st.session_state.riwayat_asesmen:
        st.markdown("## 📊 Riwayat Asesmen")

        # Admin bisa melihat semua riwayat
        if st.session_state.role == "admin":
            hasil_df = pd.DataFrame(st.session_state.riwayat_asesmen)
        else:
            # Asesor hanya melihat hasil miliknya
            hasil_df = pd.DataFrame([r for r in st.session_state.riwayat_asesmen if r["Asesor"] == st.session_state.username])

        st.dataframe(hasil_df, use_container_width=True)

        # Tombol unduh CSV
        st.download_button(
            label="💾 Simpan Riwayat sebagai CSV",
            data=hasil_df.to_csv(index=False).encode('utf-8'),
            file_name="riwayat_asesmen.csv",
            mime="text/csv"
        )

        # Tombol unduh Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            hasil_df.to_excel(writer, index=False, sheet_name='Riwayat')
        st.download_button(
            label="💾 Simpan Riwayat sebagai Excel",
            data=excel_buffer.getvalue(),
            file_name="riwayat_asesmen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Tombol reset riwayat (hanya admin)
        if st.session_state.role == "admin":
            if st.button("🔄 Reset Semua Riwayat"):
                st.session_state.riwayat_asesmen = []
                st.success("Riwayat asesmen berhasil direset.")

# Tombol logout
    if st.button("Logout"):
        logout()
