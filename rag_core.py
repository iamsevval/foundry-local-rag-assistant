import os
# pyrefly: ignore [missing-import]
from foundry_local_sdk import Configuration, FoundryLocalManager
from sentence_transformers import SentenceTransformer
import vector_db
import document_processor

_chat_model = None
_embedding_model = None

def init_systems():
    """Modelleri ve DB'yi başlatır."""
    global _chat_model, _embedding_model
    if _chat_model is None:
        print("Chat Modeli (phi-3.5) yükleniyor...")
        cache_dir = os.path.expanduser("~/.foundry/cache/models")
        config = Configuration(app_name="rag_assistant", model_cache_dir=cache_dir)
        
        if FoundryLocalManager.instance is None:
            FoundryLocalManager.initialize(config)
            
        manager = FoundryLocalManager.instance
        chat_model = manager.catalog.get_model("phi-3.5-mini")
        chat_model.load()
        _chat_model = chat_model
        
    if _embedding_model is None:
        print("Embedding Modeli (all-MiniLM-L6-v2) yükleniyor...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    vector_db.init_db()

def process_and_store_document(file_path: str):
    """Dokümanı parçalar, vektörleştirir ve SQLite'a kaydeder."""
    init_systems()
    file_name = os.path.basename(file_path)
    print(f"{file_name} işleniyor...")
    
    chunks = document_processor.process_and_chunk_file(file_path)
    for chunk in chunks:
        emb = _embedding_model.encode([chunk])[0].tolist()
        vector_db.insert_chunk(file_name, chunk, emb)
        
    return len(chunks)

def generate_answer_stream(messages, context="", custom_persona=None):
    """
    Streamlit arayüzüne kelime kelime akacak (streaming) şekilde cevap üretir.
    custom_persona ile modelin karakteri anlık olarak değiştirilebilir.
    """
    init_systems()
    
    # Varsayılan veya özel karakter
    if custom_persona and custom_persona.strip():
        system_prompt = custom_persona.strip()
    else:
        system_prompt = (
            "Sen güvenilir, yerel ve gizlilik odaklı bir AI asistanısın. "
            "Sana sağlanan BAĞLAM (Context) metnini kullanarak kullanıcının sorusunu cevapla. "
            "Eğer sorunun cevabı BAĞLAM içerisinde YOKSA, sadece 'Bilmiyorum' de, kendi bilginle uydurma yapma."
        )
    
    last_user_msg = messages[-1]["content"]
    injected_msg = f"BAĞLAM:\n{context}\n\nSORU:\n{last_user_msg}"
    
    final_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages[:-1]:
        final_messages.append(msg)
    final_messages.append({"role": "user", "content": injected_msg})
    
    client = _chat_model.get_chat_client()
    for chunk in client.complete_streaming_chat(final_messages):
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def retrieve_context(question: str, top_k: int = 3):
    """Soruya en uygun parçaları Hem Kelime Hem Vektör (Hibrit) aramasıyla getirir."""
    init_systems()
    q_emb = _embedding_model.encode([question])[0].tolist()
    # Artık FTS5 + Vektör RRF aramasını kullanıyoruz
    results = vector_db.search_hybrid(question, q_emb, top_k)
    return results

def clear_db():
    """Tüm veritabanını sıfırlar."""
    vector_db.clear_database()

def get_document_list():
    """Yüklenmiş dosyaların listesini döndürür."""
    return vector_db.get_loaded_documents()
