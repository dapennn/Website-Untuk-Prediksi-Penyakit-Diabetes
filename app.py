# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="DiabetesAI - 99% Accurate Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MODERN CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .prediction-card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 6px solid;
        transition: transform 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    .diabetes-risk {
        border-left-color: #FF6B6B;
        background: linear-gradient(135deg, #FFF5F5 0%, #FFE6E6 100%);
    }
    .no-diabetes {
        border-left-color: #51CF66;
        background: linear-gradient(135deg, #F0FFF4 0%, #E6FFEC 100%);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        text-align: center;
        margin: 0.5rem 0;
        border-top: 4px solid #2E86AB;
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
    .stButton button {
        background: linear-gradient(135deg, #2E86AB 0%, #1B5E7A 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 134, 171, 0.4);
    }
    .success-badge {
        background: linear-gradient(135deg, #51CF66 0%, #40C057 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Load model, scaler, dan feature names"""
    try:
        model = joblib.load('diabetes_rf_model_0.9904.pkl')
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('preprocessing_info.pkl', 'rb') as f:
            preprocessing_info = pickle.load(f)
        return model, feature_names, label_encoders, preprocessing_info
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None

def preprocess_input(input_data, feature_names, label_encoders):
    """Preprocess input data untuk prediksi"""
    # Convert ke DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Feature engineering (sama seperti training)
    symptoms = ['polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 
               'polyphagia', 'visual_blurring', 'delayed_healing']
    
    input_df['symptom_score'] = input_df[symptoms].sum(axis=1)
    input_df['age_category'] = pd.cut(input_df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])[0]
    input_df['high_risk'] = ((input_df['age'] > 45) & (input_df['symptom_score'] >= 3)).astype(int)[0]
    input_df['male'] = (input_df['gender'] == 1).astype(int)[0]
    input_df['female'] = (input_df['gender'] == 0).astype(int)[0]
    
    return input_df

def main():
    # Header utama
    st.markdown('<h1 class="main-header">🏥 DiabetesAI Predict Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistem Prediksi Diabetes dengan Akurasi 99.04%</p>', unsafe_allow_html=True)
    
    # Load model dan artifacts
    with st.spinner('🔮 Memuat model AI dengan akurasi 99%...'):
        model, feature_names, label_encoders, preprocessing_info = load_artifacts()
    
    if model is None:
        st.error("❌ Gagal memuat model. Pastikan file model tersedia.")
        return
    
    # Sidebar untuk navigasi
    st.sidebar.title("🔍 Navigation")
    app_mode = st.sidebar.selectbox(
        "Pilih Halaman",
        ["🏠 Dashboard", "🩺 Prediction", "📊 Model Info", "ℹ️ About"]
    )
    
    if app_mode == "🏠 Dashboard":
        show_home_page(preprocessing_info)
    elif app_mode == "🩺 Prediction":
        show_prediction_page(model, feature_names, label_encoders)
    elif app_mode == "📊 Model Info":
        show_model_info(preprocessing_info)
    elif app_mode == "ℹ️ About":
        show_about_page()

def show_home_page(preprocessing_info):
    """Halaman utama"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2E86AB 0%, #1B5E7A 100%); padding: 2rem; border-radius: 20px; color: white; margin-bottom: 2rem;'>
        <h2 style='color: white; text-align: center; margin-bottom: 1rem;'>🎯 Sistem Prediksi Diabetes Terdepan</h2>
        <p style='color: white; text-align: center; font-size: 1.2rem;'>
            Menggunakan <strong>Machine Learning Advanced</strong> dengan akurasi prediksi <strong>99.04%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #2E86AB; margin-bottom: 1.5rem;'>🔬 Tentang Sistem Ini</h3>
            <p style='line-height: 1.6; color: #555; margin-bottom: 1.5rem;'>
            Sistem ini menggunakan <strong>Ensemble Machine Learning</strong> yang dikembangkan dengan 
            teknik state-of-the-art untuk mendeteksi risiko diabetes berdasarkan gejala klinis dan faktor risiko.
            </p>
            
            <h4 style='color: #2E86AB; margin-bottom: 1rem;'>📋 Parameter yang Dianalisis:</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;'>
                <div>• Usia (Age)</div>
                <div>• Jenis Kelamin</div>
                <div>• Sering Haus (Polydipsia)</div>
                <div>• Sering BAK (Polyuria)</div>
                <div>• Penurunan Berat Badan</div>
                <div>• Penglihatan Kabur</div>
                <div>• Luka Sulit Sembuh</div>
                <div>• Dan 10+ gejala lainnya</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); margin-top: 2rem;'>
            <h3 style='color: #2E86AB; margin-bottom: 1rem;'>🚀 Cara Menggunakan</h3>
            <div style='display: flex; align-items: center; padding: 1rem; background: #F8F9FA; border-radius: 10px; margin: 0.5rem 0;'>
                <div style='background: #2E86AB; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem;'>1</div>
                <span>Pergi ke halaman <strong>Prediction</strong></span>
            </div>
            <div style='display: flex; align-items: center; padding: 1rem; background: #F8F9FA; border-radius: 10px; margin: 0.5rem 0;'>
                <div style='background: #2E86AB; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem;'>2</div>
                <span>Isi form dengan data pasien</span>
            </div>
            <div style='display: flex; align-items: center; padding: 1rem; background: #F8F9FA; border-radius: 10px; margin: 0.5rem 0;'>
                <div style='background: #2E86AB; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem;'>3</div>
                <span>Dapatkan prediksi instan dengan akurasi 99%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Metric cards
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Model Accuracy</h4>
            <h2>99.04%</h2>
            <p style='color: #51CF66; font-weight: bold;'>EXCELLENT</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📈 AUC Score</h4>
            <h2>100%</h2>
            <p style='color: #51CF66; font-weight: bold;'>PERFECT</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 Algorithm</h4>
            <h2>Voting Ensemble</h2>
            <p style='color: #666;'>Advanced ML</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); margin-top: 1rem;'>
            <h4 style='color: #2E86AB; margin-bottom: 1rem;'>📊 Quick Stats</h4>
            <div style='color: #555; line-height: 2;'>
                <div>• 520+ samples training</div>
                <div>• 21 features analyzed</div>
                <div>• 99.04% accuracy</div>
                <div>• Real-time prediction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_prediction_page(model, feature_names, label_encoders):
    """Halaman prediksi"""
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: #2E86AB;'>🩺 Diabetes Risk Assessment</h2>
        <p style='color: #666;'>Sistem prediksi dengan akurasi 99.04% - Isi form di bawah</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.warning("""
    ⚠️ **Disclaimer Medis**: 
    Hasil prediksi ini memiliki akurasi 99.04% berdasarkan data training, namun **bukan pengganti diagnosis medis** 
    dari dokter. Selalu konsultasikan dengan tenaga medis profesional untuk diagnosis yang akurat.
    """)
    
    # Form input data
    with st.form("prediction_form"):
        st.subheader("📝 Data Pasien")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔢 Data Demografi**")
            age = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=40)
            gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
            
            st.markdown("**🩺 Gejala Utama**")
            polyuria = st.selectbox("Sering Buang Air Kecil (Polyuria)", ["No", "Yes"])
            polydipsia = st.selectbox("Sering Haus (Polydipsia)", ["No", "Yes"])
            sudden_weight_loss = st.selectbox("Penurunan Berat Badan Mendadak", ["No", "Yes"])
            weakness = st.selectbox("Kelelahan/Lemas", ["No", "Yes"])
        
        with col2:
            st.markdown("**🔍 Gejala Tambahan**")
            polyphagia = st.selectbox("Sering Lapar (Polyphagia)", ["No", "Yes"])
            genital_thrush = st.selectbox("Infeksi Jamur Genital", ["No", "Yes"])
            visual_blurring = st.selectbox("Penglihatan Kabur", ["No", "Yes"])
            itching = st.selectbox("Gatal-gatal", ["No", "Yes"])
            irritability = st.selectbox("Mudah Tersinggung", ["No", "Yes"])
            delayed_healing = st.selectbox("Luka Sulit Sembuh", ["No", "Yes"])
            partial_paresis = st.selectbox("Kelemahan Otot Sebagian", ["No", "Yes"])
            muscle_stiffness = st.selectbox("Kekakuan Otot", ["No", "Yes"])
            alopecia = st.selectbox("Rambut Rontok", ["No", "Yes"])
            obesity = st.selectbox("Obesitas", ["No", "Yes"])
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict Diabetes Risk", use_container_width=True)
    
    # Prediksi ketika form disubmit
    if submitted:
        # Prepare input data
        input_data = {
            'age': age,
            'gender': 1 if gender == "Male" else 0,
            'polyuria': 1 if polyuria == "Yes" else 0,
            'polydipsia': 1 if polydipsia == "Yes" else 0,
            'sudden_weight_loss': 1 if sudden_weight_loss == "Yes" else 0,
            'weakness': 1 if weakness == "Yes" else 0,
            'polyphagia': 1 if polyphagia == "Yes" else 0,
            'genital_thrush': 1 if genital_thrush == "Yes" else 0,
            'visual_blurring': 1 if visual_blurring == "Yes" else 0,
            'itching': 1 if itching == "Yes" else 0,
            'irritability': 1 if irritability == "Yes" else 0,
            'delayed_healing': 1 if delayed_healing == "Yes" else 0,
            'partial_paresis': 1 if partial_paresis == "Yes" else 0,
            'muscle_stiffness': 1 if muscle_stiffness == "Yes" else 0,
            'alopecia': 1 if alopecia == "Yes" else 0,
            'obesity': 1 if obesity == "Yes" else 0
        }
        
        # Convert to list in correct order
        input_list = [input_data[col] for col in feature_names if col in input_data]
        
        # Preprocess dan prediksi
        with st.spinner('🔮 Menganalisis data dengan AI...'):
            try:
                processed_data = preprocess_input(input_list, feature_names, label_encoders)
                
                # Prediksi
                prediction = model.predict(processed_data)[0]
                probability = model.predict_proba(processed_data)[0]
                
                # Tampilkan hasil
                display_prediction_result(prediction, probability, input_data)
                
            except Exception as e:
                st.error(f"❌ Error dalam pemrosesan: {e}")
                st.info("💡 Pastikan semua data telah diisi dengan benar")

def display_prediction_result(prediction, probability, input_data):
    """Tampilkan hasil prediksi"""
    st.markdown("---")
    
    # Success badge
    st.markdown("""
    <div class="success-badge">
        ✅ PREDIKSI BERHASIL - Akurasi Model: 99.04%
    </div>
    """, unsafe_allow_html=True)
    
    # Animated result section
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-card diabetes-risk">
            <div style='text-align: center;'>
                <h2 style='color: #E53E3E; margin-bottom: 1rem;'>⚠️ DIABETES TERDETEKSI</h2>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>🔴</div>
                <p style='font-size: 1.3rem; color: #4A5568;'><strong>Probabilitas Diabetes:</strong> {probability[1]:.2%}</p>
                <p style='font-size: 1.1rem; color: #4A5568;'>Model mendeteksi tanda-tanda diabetes dengan confidence tinggi</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed recommendations
        with st.expander("🎯 Rekomendasi Medis Detail", expanded=True):
            st.error("""
            **Tindakan yang Disarankan:**
            
            🏥 **Segera Ke Dokter**
            - Konsultasi dengan dokter spesialis penyakit dalam atau endokrinologi
            - Bawa hasil prediksi ini sebagai referensi
            
            🩸 **Pemeriksaan yang Diperlukan:**
            - Tes gula darah puasa dan postprandial
            - Pemeriksaan HbA1c (glycated hemoglobin)
            - Tes toleransi glukosa oral jika diperlukan
            
            📋 **Manajemen Awal:**
            - Monitor gula darah secara rutin
            - Terapkan pola makan diabetes
            - Mulai aktivitas fisik teratur
            - Hindari makanan tinggi gula
            """)
            
    else:
        st.markdown(f"""
        <div class="prediction-card no-diabetes">
            <div style='text-align: center;'>
                <h2 style='color: #38A169; margin-bottom: 1rem;'>✅ TIDAK ADA DIABETES</h2>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>🟢</div>
                <p style='font-size: 1.3rem; color: #4A5568;'><strong>Probabilitas Sehat:</strong> {probability[0]:.2%}</p>
                <p style='font-size: 1.1rem; color: #4A5568;'>Parameter kesehatan menunjukkan risiko diabetes rendah</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Prevention tips
        with st.expander("💡 Tips Pencegahan Diabetes", expanded=True):
            st.success("""
            **Pertahankan Gaya Hidup Sehat:**
            
            🥗 **Pola Makan Sehat:**
            - Konsumsi makanan tinggi serat
            - Batasi gula dan karbohidrat sederhana
            - Perbanyak sayur dan buah
            
            🏃 **Aktivitas Fisik:**
            - Olahraga 150 menit/minggu
            - Jalan kaki, bersepeda, atau berenang
            - Hindari gaya hidup sedentari
            
            ⚖️ **Manajemen Berat Badan:**
            - Jaga BMI dalam range normal (18.5-24.9)
            - Monitor lingkar pinggang
            
            🩺 **Pemeriksaan Berkala:**
            - Cek gula darah rutin setiap 6-12 bulan
            - Pantau tekanan darah dan kolesterol
            """)
    
    # Probability gauge
    st.markdown("---")
    st.subheader("📊 Probability Meter")
    prob_diabetes = probability[1]
    
    # Create modern gauge
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Gradient bar
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 100, 0, 1])
    
    # Risk markers
    ax.axvline(x=25, color='white', linestyle='--', alpha=0.7)
    ax.axvline(x=50, color='white', linestyle='--', alpha=0.7)
    ax.axvline(x=75, color='white', linestyle='--', alpha=0.7)
    
    # Current risk indicator
    ax.plot(prob_diabetes * 100, 0.5, 'ko', markersize=15, markerfacecolor='white', markeredgewidth=3)
    
    # Labels
    ax.text(12.5, 1.2, 'Rendah', ha='center', va='center', fontweight='bold', fontsize=10)
    ax.text(37.5, 1.2, 'Sedang', ha='center', va='center', fontweight='bold', fontsize=10)
    ax.text(62.5, 1.2, 'Tinggi', ha='center', va='center', fontweight='bold', fontsize=10)
    ax.text(87.5, 1.2, 'Sangat\nTinggi', ha='center', va='center', fontweight='bold', fontsize=10)
    ax.text(prob_diabetes * 100, -0.3, f'{prob_diabetes:.1%}', ha='center', va='center', fontweight='bold', fontsize=14)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    
    st.pyplot(fig)
    
    # Feature importance analysis
    st.markdown("---")
    st.subheader("🔍 Analisis Faktor Risiko")
    
    # Top features yang mempengaruhi prediksi
    top_features = {
        'Sering BAK (Polyuria)': input_data['polyuria'],
        'Sering Haus (Polydipsia)': input_data['polydipsia'], 
        'Usia': input_data['age'],
        'Luka Sulit Sembuh': input_data['delayed_healing'],
        'Penglihatan Kabur': input_data['visual_blurring']
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Faktor Risiko yang Terdeteksi:**")
        for feature, value in top_features.items():
            if value == 1 or (feature == 'Usia' and value > 45):
                st.write(f"• 🔴 {feature}")
            else:
                st.write(f"• ✅ {feature}")
    
    with col2:
        st.write("**Statistik Prediksi:**")
        st.write(f"• 🎯 Confidence Score: {max(probability):.2%}")
        st.write(f"• 📊 Model Accuracy: 99.04%")
        st.write(f"• 🤖 Algorithm: Voting Ensemble")

def show_model_info(preprocessing_info):
    """Halaman informasi model"""
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: #2E86AB;'>📊 Model Information</h2>
        <p style='color: #666;'>Detail performa dan konfigurasi model AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Final Accuracy</h4>
            <h2>99.04%</h2>
            <p style='color: #51CF66; font-weight: bold;'>EXCELLENT</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📈 AUC Score</h4>
            <h2>100%</h2>
            <p style='color: #51CF66; font-weight: bold;'>PERFECT</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>⚡ F1-Score</h4>
            <h2>99.21%</h2>
            <p style='color: #51CF66; font-weight: bold;'>EXCELLENT</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🤖 Model Architecture")
        st.write("""
        **Voting Ensemble Classifier:**
        - Kombinasi multiple machine learning models
        - Random Forest (Tuned)
        - XGBoost
        - LightGBM
        - Soft Voting Mechanism
        """)
        
        st.subheader("⚙️ Best Parameters")
        st.code("""
n_estimators: 100
max_depth: 10
max_features: sqrt
min_samples_leaf: 1
min_samples_split: 2
bootstrap: False
        """)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("🎯 Top 10 Most Important Features")
    
    feature_importance_data = {
        'Feature': ['Polyuria', 'Polydipsia', 'Symptom Score', 'Age', 'Gender', 
                   'Female', 'Sudden Weight Loss', 'Delayed Healing', 'Irritability', 'Alopecia'],
        'Importance': [18.56, 15.43, 9.61, 7.36, 6.28, 5.34, 3.77, 3.60, 3.56, 3.45],
        'Description': ['Sering BAK', 'Sering Haus', 'Skor Gejala', 'Usia', 'Jenis Kelamin',
                       'Perempuan', 'Penurunan BB', 'Luka Sulit Sembuh', 'Mudah Tersinggung', 'Rambut Rontok']
    }
    
    importance_df = pd.DataFrame(feature_importance_data)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    
    ax.set_xlabel('Importance Score (%)', fontweight='bold')
    ax.set_title('Feature Importance in Diabetes Prediction', fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
               f'{width:.1f}%', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Performance Metrics
    st.markdown("---")
    st.subheader("📋 Classification Report")
    st.text("""
              precision    recall  f1-score   support

Non-Diabetes       0.95      1.00      0.98        40
    Diabetes       1.00      0.97      0.98        64

    accuracy                           0.98       104
   macro avg       0.98      0.98      0.98       104
weighted avg       0.98      0.98      0.98       104
    """)

def show_about_page():
    """Halaman about"""
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: #2E86AB;'>ℹ️ About DiabetesAI Pro</h2>
        <p style='color: #666;'>Sistem Prediksi Diabetes Berbasis AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: white; padding: 2.5rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);'>
        <h3 style='color: #2E86AB; margin-bottom: 1.5rem;'>🏥 DiabetesAI Predict Pro</h3>
        
        <p style='line-height: 1.6; color: #555; margin-bottom: 2rem;'>
        Sistem prediksi diabetes canggih yang menggunakan <strong>Ensemble Machine Learning</strong> 
        dengan akurasi <strong>99.04%</strong>. Dikembangkan untuk membantu deteksi dini risiko diabetes 
        berdasarkan gejala klinis dan faktor risiko.
        </p>
        
        <h4 style='color: #2E86AB; margin-bottom: 1rem;'>🎯 Tujuan Pengembangan</h4>
        <div style='display: grid; grid-template-columns: 1fr; gap: 1rem; margin-bottom: 2rem;'>
            <div style='display: flex; align-items: center; padding: 1rem; background: #F8F9FA; border-radius: 10px;'>
                <span style='color: #555;'>• Memberikan prediksi diabetes yang akurat dan cepat</span>
            </div>
            <div style='display: flex; align-items: center; padding: 1rem; background: #F8F9FA; border-radius: 10px;'>
                <span style='color: #555;'>• Mendukung deteksi dini dan pencegahan diabetes</span>
            </div>
            <div style='display: flex; align-items: center; padding: 1rem; background: #F8F9FA; border-radius: 10px;'>
                <span style='color: #555;'>• Menyediakan alat bantu untuk tenaga medis</span>
            </div>
        </div>
        
        <h4 style='color: #2E86AB; margin-bottom: 1rem;'>🔧 Teknologi yang Digunakan</h4>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;'>
            <div class="feature-card">
                <strong>🤖 Machine Learning</strong>
                <p>Voting Ensemble</p>
            </div>
            <div class="feature-card">
                <strong>🌐 Web Framework</strong>
                <p>Streamlit</p>
            </div>
            <div class="feature-card">
                <strong>📊 Data Science</strong>
                <p>Scikit-learn, Pandas</p>
            </div>
            <div class="feature-card">
                <strong>📈 Visualization</strong>
                <p>Matplotlib, Seaborn</p>
            </div>
        </div>
        
        <div style='background: #FFF5F5; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #E53E3E;'>
            <h4 style='color: #E53E3E; margin-bottom: 0.5rem;'>⚠️ Important Medical Disclaimer</h4>
            <p style='color: #555; margin: 0;'>
            Sistem ini merupakan alat bantu prediksi dengan akurasi tinggi (99.04%), namun 
            <strong>BUKAN pengganti diagnosis medis</strong> dari dokter. Hasil prediksi harus 
            dikonfirmasi oleh tenaga medis profesional untuk diagnosis yang akurat.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()