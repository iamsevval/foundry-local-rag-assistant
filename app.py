# pyrefly: ignore [missing-import]
import streamlit as st
import os
import rag_core
import tempfile

st.set_page_config(page_title="RAG Asistanı", page_icon="🤖", layout="wide")

st.title("🤖 Yerel Hibrit RAG Asistanı")
st.caption("Verileriniz %100 cihazınızda kalır.")

# Sohbet Geçmişi (Memory) Başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []

# Yan menü: PDF yükleme, Ayarlar, Veritabanı Yönetimi
with st.sidebar:
    st.header("📂 Dosya Yükle")
    uploaded_file = st.file_uploader("PDF veya TXT seçin", type=["pdf", "txt"])
    
    if st.button("Veritabanına Ekle"):
        if uploaded_file is not None:
            with st.spinner("Dosya işleniyor (Hibrit İndeksleniyor)..."):
                ext = uploaded_file.name.split('.')[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                target_dir = os.path.join(os.getcwd(), "uploads")
                os.makedirs(target_dir, exist_ok=True)
                final_path = os.path.join(target_dir, uploaded_file.name)
                
                with open(final_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                chunk_count = rag_core.process_and_store_document(final_path)
                st.success(f"Başarılı! '{uploaded_file.name}' dosyasından {chunk_count} parça SQLite veritabanına indekslendi.")
        else:
            st.warning("Lütfen bir dosya seçin.")

    st.divider()
    
    st.header("🎭 Asistan Karakteri")
    persona_input = st.text_area(
        "Modelin nasıl davranmasını istiyorsunuz?", 
        value="Sen güvenilir, yerel ve gizlilik odaklı bir AI asistanısın. Sana sağlanan BAĞLAM metnini kullanarak kullanıcının sorusunu cevapla. Cevap BAĞLAM'da yoksa sadece 'Bilmiyorum' de.",
        height=150
    )

    st.divider()

    st.header("🗄️ Veritabanı Yönetimi")
    
    # Yüklü dosyaları listele
    try:
        loaded_docs = rag_core.get_document_list()
        if loaded_docs:
            st.write("**Yüklü Dosyalar:**")
            for doc in loaded_docs:
                st.caption(f"- {doc}")
        else:
            st.write("*Veritabanı şu an boş.*")
    except Exception:
        pass

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sohbeti Temizle"):
            st.session_state.messages = []
            st.rerun()
            
    with col2:
        if st.button("Veritabanını Sıfırla"):
            rag_core.clear_db()
            st.session_state.messages = []
            st.success("Tüm indeksler silindi!")
            st.rerun()

# Önceki mesajları ekranda göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Kaynaklar (Hibrit Skorlu)"):
                for src in message["sources"]:
                    st.caption(f"**{src['source']}** - (RRF Skoru: {src['distance']:.4f})\n\n{src['content']}")

# Yeni Soru Girişi
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # RAG: Veritabanında ara (Hibrit: FTS5 + Vec)
    with st.spinner("Vektör ve Kelime aramaları harmanlanıyor..."):
        results = rag_core.retrieve_context(prompt, top_k=7)
        
    context_str = ""
    for i, r in enumerate(results):
        context_str += f"[KAYNAK {i+1} - {r['source']}]: {r['content']}\n\n"
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Seçilen Persona ile akıcı cevap üret
        stream = rag_core.generate_answer_stream(
            st.session_state.messages, 
            context=context_str, 
            custom_persona=persona_input
        )
        
        for chunk in stream:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        
        if results:
            with st.expander("Kullanılan Kaynaklar (Hibrit Skorlu)"):
                for r in results:
                    st.caption(f"**Dosya:** {r['source']} (RRF Skoru: {r['distance']:.4f})\n> {r['content']}")
                    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": results
    })
