from typing import Optional, Dict, Any, Tuple

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
import io
import os
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Konfigurasikan eager execution
tf.config.run_functions_eagerly(True)

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Motif Batik Yogyakarta",
    page_icon="👕",
    layout="wide"
)

# Judul aplikasi
st.title("Klasifikasi Motif Batik Keraton Yogyakarta")
st.markdown("Aplikasi ini menggunakan model CNN AlexNet yang dioptimasi dengan PSO untuk klasifikasi motif batik Yogyakarta.")

@st.cache_resource
def load_colab_model(model_path):
    try:
        # Standard mode: Keras load_model
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        return None

# Fungsi untuk preprocessing gambar
def preprocess_image(image, target_size=(227, 227)):
    # Resize gambar sesuai dengan ukuran input AlexNet (227x227 untuk AlexNet original)
    image = image.resize(target_size)
    # Konversi ke array
    image_array = np.array(image)
    
    # Handle gambar grayscale (1 channel) - konversi ke RGB
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,)*3, axis=-1)
    # Handle gambar RGBA (4 channels) - konversi ke RGB
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    
    # Normalisasi (untuk AlexNet biasanya nilai piksel dalam rentang 0-1)
    image_array = image_array / 255.0
    
    # Expand dimensi untuk batch
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

class_names = ["Kawung", "Parang", "Truntum"]

# Fungsi prediksi
def predict(model, image):
    try:
        # Preprocessing gambar
        processed_image = preprocess_image(image)
        
        # Debug informasi
        print(f"Input shape: {processed_image.shape}")
        print(f"Model type: {type(model)}")
        
        # Coba prediksi dengan berbagai metode
        try:
            # Mode 1: Keras model
            if hasattr(model, 'predict'):
                predictions = model.predict(processed_image)
                
                # Log confidence
                confidences = predictions[0]
                for i, conf in enumerate(confidences):
                    print(f"Confidence for {class_names[i]}: {conf * 100:.2f}%")
                
                # Pastikan predictions berbentuk array 1D atau 2D dengan panjang sesuai kelas
                predictions = np.array(predictions)
                if predictions.ndim > 1:
                    predictions = predictions[0]
                
                print(f"Predictions shape: {predictions.shape}")
                return predictions
        
        except Exception as e:
            st.error(f"Error prediksi utama: {e}")
            
            # Mode 3: Alternate prediction method
            try:
                # Convert to TensorFlow tensor
                tf_image = tf.convert_to_tensor(processed_image, dtype=tf.float32)
                
                # Gunakan fungsi invoke jika SavedModel
                if hasattr(model, '__call__'):
                    predictions = model(tf_image, training=False).numpy()
                    
                    # Pastikan predictions berbentuk array 1D atau 2D dengan panjang sesuai kelas
                    predictions = np.array(predictions)
                    if predictions.ndim > 1:
                        predictions = predictions[0]
                    
                    print(f"Predictions shape: {predictions.shape}")
                    return predictions
            
            except Exception as e_alt:
                st.error(f"Error prediksi alternatif: {e_alt}")
                return None
    
    except Exception as e:
        st.error(f"Error fatal saat prediksi: {e}")
        return None

# Informasi motif batik
def get_batik_info(motif_name):
    batik_info = {
        "Kawung": {
            "desc": "Motif Kawung terdiri dari bentuk-bentuk bulat yang saling bersinggungan, menyerupai buah kawung (sejenis buah aren).",
            "origin": "Yogyakarta",
            "history": "Motif Kawung merupakan salah satu motif tertua yang berasal dari era Kerajaan Mataram Kuno. Pada masa lalu, motif ini hanya boleh dipakai oleh kalangan kerajaan saja.",
            "meaning": "Melambangkan harapan agar pemakainya menjadi manusia unggul dan hidupnya penuh makna."
        },
        "Parang": {
            "desc": "Motif Parang terdiri dari bentuk-bentuk yang disusun diagonal, terinspirasi dari karang di laut yang terkena ombak.",
            "origin": "Yogyakarta",
            "history": "Motif Parang hanya boleh dikenakan oleh raja, bangsawan, dan kesatria.",
            "meaning": "Parang mencerminkan nilai-nilai kepemimpinan dan keteguhan, sekaligus menjadi simbol perlawanan terhadap kejahatan melalui pengendalian diri."
        },
        "Truntum": {
            "desc": "Motif Truntum terdiri dari gambar bunga-bunga kecil menyerupai bintang dengan delapan kelopak.",
            "origin": "Yogyakarta",
            "history": "Motif Truntum diciptakan oleh Kanjeng Ratu Beruk, istri dari Pakubuwono III.",
            "meaning": "Melambangkan cinta yang tumbuh kembali, kesetiaan dan ketulusan."
        }
    }
    return batik_info.get(motif_name, {})

# Fungsi untuk menampilkan hasil klasifikasi
# Fungsi untuk menampilkan hasil klasifikasi
def display_prediction_result(predictions, class_names, image):
    # Pastikan predictions adalah numpy array
    predictions = np.array(predictions)
    
    # Tambahkan threshold
    threshold = 0.6  # Batas kepercayaan prediksi
    
    # Debug print
    print("Raw Predictions:", predictions)
    print("Predictions shape:", predictions.shape)
    
    # Pastikan predictions memiliki bentuk yang benar
    if predictions.ndim > 1:
        predictions = predictions[0]
    
    # Cek threshold
    max_confidence = np.max(predictions)
    
    # Pastikan panjang predictions sesuai dengan class_names
    if len(predictions) != len(class_names):
        st.error(f"Panjang predictions ({len(predictions)}) tidak sesuai dengan jumlah kelas ({len(class_names)})")
        return
    
    # Mendapatkan indeks kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[predicted_class_index] * 100
    
    # Tambahkan peringatan jika confidence rendah
    if max_confidence < threshold:
        st.warning(f"⚠️ Klasifikasi tidak yakin. Confidence hanya {confidence:.2f}%")
    # Tampilkan hasil dalam 1 kolom
    col1, = st.columns(1)

    with col1:
        st.subheader("Hasil Prediksi")
        st.markdown(f"**Motif Batik:** {predicted_class}")
        st.markdown(f"**Tingkat Keyakinan:** {confidence:.2f}%")
    
    # Tampilkan informasi detail tentang motif batik
    st.subheader(f"Tentang Motif {predicted_class}")
    
    # Dapatkan informasi motif batik
    info = get_batik_info(predicted_class)
    
    if info:
        st.markdown(f"**Deskripsi:** {info['desc']}")
        
        # Tampilkan informasi dalam 3 kolom
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Asal:**")
            st.info(info['origin'])
        
        with col2:
            st.info(f"**Makna:**")
            st.info(info['meaning'])
        with col3:
            if 'history' in info:
                st.success("**Sejarah:**")
                st.success(info['history'])
    else:
        st.write("Informasi detail tentang motif ini belum tersedia.")

    # Di bagian sidebar, tambahkan detail confidence
    st.sidebar.title("Detail Confidence")
    for i, (name, conf) in enumerate(zip(class_names, predictions * 100)):
        st.sidebar.progress(int(conf), text=f"{name}: {conf:.2f}%")

# Main Streamlit App
def main():
    # Sidebar
    st.sidebar.title("Tentang Aplikasi")
    st.sidebar.info(
        """
        Aplikasi ini mengklasifikasikan 3 motif batik Yogyakarta:
        - Kawung
        - Parang
        - Truntum
        """
    )

    # Path model
    model_path = r"c:/Users/USER/Skripsi/Final Model Alexnet PSO.h5"

    # Coba load model
    model = load_colab_model(model_path)

    # Class names (sesuaikan dengan kelas di model Anda)
    class_names = ["Kawung", "Parang", "Truntum"]

    # Tabs untuk metode input
    tab1, = st.tabs(["Upload Gambar"])

    with tab1:
        # Upload gambar
        uploaded_file = st.file_uploader("Pilih gambar motif batik...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Baca gambar yang diunggah
            image = Image.open(uploaded_file)
            
            # Tampilkan gambar yang diunggah
            st.image(image, caption="Gambar yang Diunggah", width=300)
            
            # Tombol untuk memproses
            if st.button("Proses Gambar"):
                if model:
                    with st.spinner("Mengklasifikasikan..."):
                        # Lakukan prediksi
                        predictions = predict(model, image)
                        
                        if predictions is not None:
                            # Tampilkan hasil
                            display_prediction_result(predictions, class_names, image)
                else:
                    st.error("Model tidak dapat dimuat. Periksa path model.")

    # Footer
    st.markdown("---")

# Jalankan aplikasi
if __name__ == "__main__":
    main()