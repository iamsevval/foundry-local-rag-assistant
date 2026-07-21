# pyrefly: ignore [missing-import]
import streamlit as st
import os
import rag_core
import tempfile
from pyvis.network import Network
import streamlit.components.v1 as components
import vector_db

st.set_page_config(page_title="RAG Asistanı", page_icon="🤖", layout="wide")

st.title("🤖 Yerel Hibrit RAG Asistanı")
st.caption("Verileriniz %100 cihazınızda kalır.")

# Sohbet Geçmişi (Memory) Başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []

# Yan menü: PDF yükleme, Ayarlar, Veritabanı Yönetimi
with st.sidebar:
    st.header("📂 Dosya Yükle")
    uploaded_file = st.file_uploader("PDF, DOCX veya TXT seçin", type=["pdf", "docx", "txt"])
    
    if st.button("Veritabanına Ekle"):
        if uploaded_file is not None:
            with st.spinner("Dosya işleniyor (Hibrit İndeksleniyor)..."):
                try:
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
                except Exception as e:
                    st.error(f"Dosya yüklenirken bir hata oluştu: {str(e)}")
        else:
            st.warning("Lütfen bir dosya seçin.")

    st.divider()
    
    st.header("🎭 Asistan Karakteri")
    persona_input = st.text_area(
        "Modelin nasıl davranmasını istiyorsunuz?", 
        value="Sen anlamsal çıkarım (semantic reasoning) yeteneği yüksek bir AI asistanısın. Kullanıcının sorusunu cevaplamadan önce BAĞLAM metnini adım adım analiz et. Kavramları eşleştir (örneğin 'kriz', 'aksaklık' veya 'problem' benzer şeylerdir). Metinde olmayan isimleri/olayları ASLA uydurma, ancak var olan bilgileri eşanlamlılarıyla mantıklı bir şekilde yorumla. Cevap bağlamda yoksa 'Bilmiyorum' de.",
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
                col_doc, col_btn = st.columns([3, 1])
                with col_doc:
                    display_name = doc if len(doc) < 25 else doc[:22] + "..."
                    st.write(f"📄 {display_name}")
                with col_btn:
                    if st.button("Sil", key=f"del_{doc}", help="Dosyayı sil"):
                        rag_core.delete_document(doc)
                        st.rerun()
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

tab_chat, tab_graph = st.tabs(["💬 Sohbet", "🕸️ Bilgi Grafiği (Graph-RAG)"])

with tab_chat:
    # Önceki mesajları ekranda göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "rewritten_query" in message and message["rewritten_query"]:
                st.info(f"✍️ Düzeltilen Sorgu: {message['rewritten_query']}")
            if "sources" in message and message["sources"]:
                with st.expander("Kullanılan Kaynaklar (Re-rank Skorlu)"):
                    for src in message["sources"]:
                        score = src.get('cross_encoder_score', 0)
                        st.caption(f"**{src['source']}** - (Skor: {score:.4f})\n\n{src['content']}")

    # Yeni Soru Girişi
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # RAG: Veritabanında ara
        st.status("Arama Tamamlandı!", state="complete")
        with st.spinner("Soru zenginleştiriliyor ve veritabanı taranıyor..."):
            try:
                results = rag_core.retrieve_context(st.session_state.messages, prompt, top_k=7)
            except Exception as e:
                st.error(f"Veritabanında arama yapılırken hata oluştu: {str(e)}")
                results = []
            
        rewritten = results[0]['rewritten_query'] if results and 'rewritten_query' in results[0] else None
        if rewritten and rewritten != prompt:
            st.info(f"✍️ Düzeltilen Sorgu: {rewritten}")
            
        st.caption("Sonuçlar Cross-Encoder ile yeniden sıralandı.")
            
        context_str = ""
        for i, r in enumerate(results):
            context_str += f"[KAYNAK {i+1} - {r['source']}]: {r['content']}\n\n"
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            if not results:
                full_response = "Bilmiyorum. (Veritabanında eşleşen hiçbir bilgi bulunamadı veya hiç dosya yüklemediniz.)"
                message_placeholder.markdown(full_response)
            else:
                try:
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
                    
                    with st.expander("Kullanılan Kaynaklar (Re-rank Skorlu)"):
                        for r in results:
                            score = r.get('cross_encoder_score', 0)
                            st.caption(f"**Dosya:** {r['source']} (Skor: {score:.4f})\n> {r['content']}")
                except Exception as e:
                    st.error(f"Cevap üretilirken yapay zeka hatası oluştu: {str(e)}")
                        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "sources": results,
            "rewritten_query": rewritten if rewritten != prompt else None
        })

with tab_graph:
    st.subheader("🕸️ Graph-RAG Haritası")
    st.caption("Belgelerden otomatik çıkarılan varlıklar (Kişi, Proje, Teknoloji) ve aralarındaki anlamsal ilişkiler.")
    
    edges = vector_db.get_all_graph_edges()
    
    if not edges:
        st.info("Henüz ağ haritası oluşturulamadı. Lütfen yeni bir PDF yükleyin. Model ilk kısımları analiz ederek varlıkları çekecektir.")
    else:
        net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white", directed=True, cdn_resources="in_line")
        # Varlıkları ve kenarları ekle
        for edge in edges:
            src = edge.get("source_node", "Bilinmeyen")
            tgt = edge.get("target_node", "Bilinmeyen")
            rel = edge.get("relation", "")
            
            # Aynı düğüm iki kez eklense bile pyvis id bazlı otomatik ezer/bırakır
            net.add_node(src, label=src, title=src, color="#1f77b4")
            net.add_node(tgt, label=tgt, title=tgt, color="#ff7f0e")
            net.add_edge(src, tgt, title=rel, label=rel, color="#888888")
            
        net.repulsion(node_distance=150, spring_length=200)
        
        try:
            path = "/tmp/graph_rag.html"
            net.save_graph(path)
            with open(path, "r", encoding="utf-8") as f:
                html_data = f.read()
            components.html(html_data, height=620)
        except Exception as e:
            st.error(f"Grafik oluşturulurken hata oluştu: {e}")

