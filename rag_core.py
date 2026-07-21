import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# pyrefly: ignore [missing-import]
from foundry_local_sdk import Configuration, FoundryLocalManager
from sentence_transformers import SentenceTransformer, CrossEncoder
import vector_db
import document_processor
import json

_chat_model = None
_embedding_model = None
_cross_encoder = None

def init_systems():
    """Modelleri ve DB'yi başlatır."""
    global _chat_model, _embedding_model, _cross_encoder
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
        
    if _cross_encoder is None:
        print("Cross-Encoder Modeli (ms-marco-MiniLM-L-6-v2) yükleniyor...")
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    vector_db.init_db()

def process_and_store_document(file_path: str):
    """Dokümanı parçalar, vektörleştirir ve SQLite'a kaydeder."""
    init_systems()
    file_name = os.path.basename(file_path)
    print(f"{file_name} işleniyor...")
    
    chunks = document_processor.process_and_chunk_file(file_path)
    for i, chunk in enumerate(chunks):
        emb = _embedding_model.encode([chunk])[0].tolist()
        vector_db.insert_chunk(file_name, chunk, emb)
        
        # Sadece ilk 3 chunk için graf çıkar (performans için)
        if i < 3:
            extract_entities_from_chunk(chunk, file_name)
        
    return len(chunks)

def extract_entities_from_chunk(chunk: str, file_name: str):
    """Metin parçasından JSON formatında varlık ve ilişkileri çıkarır."""
    init_systems()
    
    system_prompt = (
        "Sen bir veri madenciliği asistanısın. Görevin verilen metinden kişi, kurum, proje, teknoloji veya olayları bulmak ve aralarındaki ilişkiyi sadece JSON formatında çıkarmaktır. "
        "Metin dışından HİÇBİR ŞEY uydurma. "
        "Çıktı KESİNLİKLE şu formatta, köşeli parantez içinde bir liste olmalıdır:\\n"
        '[\\n  {"source_node": "A kişisi", "target_node": "B projesi", "relation": "geliştirdi"}\\n]'
    )
    
    user_prompt = f"Metin:\\n{chunk}\\n\\nSadece JSON listesi:"
    
    client = _chat_model.get_chat_client()
    response = client.complete_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    output = response.choices[0].message.content.strip()
    
    edges = []
    try:
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()
            
        start_idx = output.find("[")
        end_idx = output.rfind("]")
        if start_idx != -1 and end_idx != -1:
            json_str = output[start_idx:end_idx+1]
            edges = json.loads(json_str)
            if isinstance(edges, list):
                vector_db.insert_graph_edges(edges, file_name)
    except Exception as e:
        print(f"JSON Parse Hatası (GraphRAG): {e}")

def rewrite_query(messages, original_query: str) -> str:
    """Sohbet geçmişine bakarak bağlamı kopuk soruları tek bir anlamlı soruya dönüştürür."""
    if len(messages) <= 1:
        return original_query
        
    history_text = ""
    for m in messages[:-1]:
        role = "Kullanıcı" if m["role"] == "user" else "Asistan"
        history_text += f"{role}: {m['content']}\n"
        
    system_prompt = (
        "Sen bir metin düzenleyicisin. Görevin, kullanıcının eksik sorusunu geçmişe bakarak dilbilgisi düzgün tek bir Türkçe soruya çevirmektir. "
        "Örnek: 'Peki o ne zaman bitti?' -> 'DermaSmart projesi ne zaman bitti?' "
        "SADECE düzeltilmiş soruyu yaz. 'eleştrom' gibi uydurma kelimeler kullanma."
    )
    
    user_prompt = f"Geçmiş:\n{history_text}\n\nOrijinal Soru: {original_query}\n\nSadece Düzeltilmiş Soru:"
    
    client = _chat_model.get_chat_client()
    response = client.complete_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    rewritten = response.choices[0].message.content.strip()
    if len(rewritten) > 150:
        return original_query
    return rewritten

def retrieve_context(messages, original_query: str, top_k: int = 3):
    """Soruya en uygun parçaları Hem Kelime Hem Vektör (Hibrit) aramasıyla ve Cross-Encoder sıralamasıyla getirir."""
    init_systems()
    
    search_query = rewrite_query(messages, original_query)
    q_emb = _embedding_model.encode([search_query])[0].tolist()
    
    hybrid_results = vector_db.search_hybrid(search_query, q_emb, top_k * 3)
    
    if not hybrid_results:
        return []
        
    pairs = [[search_query, res['content']] for res in hybrid_results]
    scores = _cross_encoder.predict(pairs)
    
    for i, res in enumerate(hybrid_results):
        res['cross_encoder_score'] = float(scores[i])
        res['rewritten_query'] = search_query 
        
    hybrid_results.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
    
    return hybrid_results[:top_k]

def generate_answer_stream(messages, context="", custom_persona=None):
    """
    Streamlit arayüzüne kelime kelime akacak (streaming) şekilde cevap üretir.
    custom_persona ile modelin karakteri anlık olarak değiştirilebilir.
    """
    init_systems()
    
    if custom_persona and custom_persona.strip():
        system_prompt = custom_persona.strip()
    else:
        system_prompt = (
            "Sen anlamsal çıkarım (semantic reasoning) yeteneği yüksek bir AI asistanısın. "
            "Kullanıcının sorusunu cevaplamadan önce BAĞLAM metnini adım adım analiz et. "
            "Kavramları zekice eşleştir (örneğin 'kriz', 'aksaklık' veya 'problem' kelimeleri aynı duruma işaret edebilir). "
            "Metinde KESİNLİKLE bulunmayan isimleri veya somut verileri uydurma, ancak var olan kavramları mantıklı bir şekilde yorumla. "
            "Eğer sorunun cevabı veya bir benzeri BAĞLAM içerisinde hiçbir şekilde YOKSA, 'Bilmiyorum' de."
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

def clear_db():
    """Tüm veritabanını sıfırlar."""
    vector_db.clear_database()

def get_document_list():
    """Yüklenmiş dosyaların listesini döndürür."""
    return vector_db.get_loaded_documents()

def delete_document(file_name: str):
    """Belirli bir dosyaya ait tüm verileri siler."""
    init_systems()
    return vector_db.delete_document_by_name(file_name)
