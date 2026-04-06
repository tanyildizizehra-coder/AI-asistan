import streamlit as st
import requests
import json  # Streaming (akış) işlemleri için eklendi
from PyPDF2 import PdfReader
import docx
import io
import re
from fpdf import FPDF

# RAG KÜTÜPHANELERİ
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Yapay Zeka Destekli Kurumsal İş Asistanı", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004a99; color: white; font-weight: bold; }
    .stDownloadButton>button { width: 100%; border-radius: 5px; background-color: #28a745; color: white; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e3136,#2e3136); color: white; }
    .css-17l2qt2 { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- PDF VE ANALİZ FONKSİYONLARI ---
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    rep = {"ş":"s", "Ş":"S", "ğ":"g", "Ğ":"G", "ç":"c", "Ç":"C", "ü":"u", "Ü":"U", "ö":"o", "Ö":"O", "ı":"i", "İ":"I"}
    for k, v in rep.items(): text = text.replace(k, v)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, txt=line.encode('latin-1', 'ignore').decode('latin-1'))
    return pdf.output(dest='S').encode('latin-1')

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def create_vector_db(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return FAISS.from_texts(chunks, get_embeddings())

def extract_text(file):
    if file.name.endswith('.pdf'):
        return "".join([p.extract_text() for p in PdfReader(file).pages])
    return "\n".join([p.text for p in docx.Document(file).paragraphs])

# GÜNCELLENEN FONKSİYON: Streaming (akış) destekli Ollama çağrısı
def ask_ollama(prompt, model="gemma2:2b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model, 
        "prompt": prompt, 
        "stream": True,  # Streaming aktif edildi
        "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 1024}
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=240) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"] # Kelime kelime üretir (yield)
    except Exception as e:
        yield f"Bağlantı Hatası: {str(e)}"

# 2. SIDEBAR (Asistan Yetenekleri Tam Liste)
with st.sidebar:
    st.title("🏛️ Kurumsal Panel")
    st.caption("Sürüm: v4.9 - Proaktif Asistan")
    
    mod = st.selectbox("🎯 Yapılacak İşlemi Seçin", [
        "Soru-Cevap (Mevzuat Sorgulama)",
        "Eski-Yeni Yönerge Karşılaştırma"
    ])
    
    st.divider()
    st.markdown("### 📜 Asistan Yetenekleri:")
    st.write("✅ **Akıllı Mevzuat Analizi**")
    st.write("✅ **Madde Atıflı Cevaplar**")
    st.write("✅ **Hızlandırılmış RAG Tarama**")
    st.write("✅ **Otomatik PDF Raporlama**")
    st.write("✅ **Güven Skoru Analizi**")
    st.divider()
    

# 3. ANA EKRAN
st.title("🛡️ Yapay Zeka Destekli İş Asistanı")
st.write(f"**İşlem:** {mod}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 Kaynak Dokümanı Yükleyiniz")
    if mod == "Eski-Yeni Yönerge Karşılaştırma":
        f1 = st.file_uploader("1️⃣ ESKİ Versiyon", type=["pdf", "docx"], key="v1")
        f2 = st.file_uploader("2️⃣ YENİ Versiyon", type=["pdf", "docx"], key="v2")
    else:
        uploaded_file = st.file_uploader("Analiz edilecek dosyayı seçin", type=["pdf", "docx"])
        user_query = ""
        if mod == "Soru-Cevap (Mevzuat Sorgulama)":
            user_query = st.text_input("Mevzuatla ilgili sorunuzu yazın:", placeholder="Örn: AKTS değerleri hangi maddede düzenlenmiştir?")

with col2:
            st.subheader("📝 Asistan Cevabı")
            if st.button("İŞLEMİ BAŞLAT"):
                final_prompt = "" # Her işlemde sıfırla
                
                # --- MOD 1: KARŞILAŞTIRMA ---
                if mod == "Eski-Yeni Yönerge Karşılaştırma":
                    if f1 and f2:
                        with st.spinner("İki yönerge kıyaslanıyor..."):
                            t1, t2 = extract_text(f1), extract_text(f2)
                            final_prompt = f"""Bir mevzuat denetçisi gibi davran ve iki metni kıyasla:
                            1. 'Kaldırılan Maddeler': ESKİ metinde olup YENİ metinde olmayanları listele.
                            2. 'Değişen Hükümler': Saat, AKTS, Kontenjan gibi teknik değişiklikleri belirt.
                            3. 'Personel Notu': Uygulamada personelin dikkat etmesi gereken 3 ana farkı yaz.
                            ÖNEMLİ: En sona MUTLAKA 'GÜVEN SKORU: %X' ekle.
                            
                            ESKİ: {t1[:2500]}
                            YENİ: {t2[:2500]}"""
                    else:
                        st.warning("⚠️ Lütfen ESKİ ve YENİ dosyaların ikisini de yükleyin!")

                # --- MOD 2: SORU-CEVAP ---
                elif mod == "Soru-Cevap (Mevzuat Sorgulama)":
                    if uploaded_file and user_query:
                        with st.spinner("Mevzuat veritabanı taranıyor..."):
                            text = extract_text(uploaded_file)
                            db = create_vector_db(text)
                            context = "\n".join([d.page_content for d in db.similarity_search(user_query, k=5)])
                            final_prompt = f"""SEN BİR MEVZUAT UZMANISIN. 
                            Aşağıdaki metindeki MADDE numaralarını ve TEKNİK verileri (AKTS, SAAT vb.) kullanarak soruyu yanıtla.
                            
                            METİN BİLGİSİ:
                            {context}
                            
                            PERSONEL SORUSU: {user_query}
                            
                            ÖNEMLİ: En sona MUTLAKA 'GÜVEN SKORU: %X' ekle."""
                    else:
                        st.warning("⚠️ Lütfen dökümanı yükleyin ve sorunuzu yazın!")

                # --- CEVAP ÜRETİMİ VE GÜVEN SKORU ---
                if final_prompt:
                    st.markdown("### Yanıt:")
                    
                    # GÜNCELLENEN KISIM: st.write_stream kullanıldı
                    cevap = st.write_stream(ask_ollama(final_prompt))
                    
                    # Güven Skoru Görsel Bar
                    match = re.search(r"GÜVEN SKORU: %(\d+)", cevap)
                    if match:
                        score = int(match.group(1))
                        st.write(f"📊 **Doğruluk Analizi:** %{score}")
                        st.progress(score / 100)
                    
                    st.download_button("📥 Analiz Raporunu İndir (PDF)", create_pdf(cevap), "analiz.pdf")
