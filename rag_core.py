import os
import math
from foundry_local_sdk import Configuration, FoundryLocalManager
from sentence_transformers import SentenceTransformer

# 1. Örnek Doküman Seti
DOCUMENTS = [
    "Microsoft Foundry Local, uygulamana gömülen hafif bir çalışma zamanıdır.",
    "Foundry Local ile kullanıcı verisi cihazdan hiç çıkmaz, gizlilik odaklıdır.",
    "RAG (Retrieval-Augmented Generation), bilgi getirme ve metin üretimi aşamalarını birleştirir.",
    "Apollo 11, 1969 yılında Ay'a iniş yapmıştır.",
    "Everest Dağı'nın yüksekliği 8.848 metredir ve dünyanın en yüksek dağıdır.",
    "Python, veri bilimi ve yapay zeka alanında en çok tercih edilen programlama dillerinden biridir.",
    "Işık hızı yaklaşık olarak saniyede 300.000 kilometredir."
]

def cosine_similarity(vec1, vec2):
    """İki vektör arasındaki kosinüs benzerliğini hesaplar (Dış kütüphane olmadan)."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def setup_models():
    """Foundry Local Manager'ı başlatır ve chat modelini yükler."""
    print("Chat Modeli yükleniyor (bu işlem birkaç saniye sürebilir)...")
    cache_dir = os.path.expanduser("~/.foundry/cache/models")
    config = Configuration(app_name="rag_assistant", model_cache_dir=cache_dir)
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Sadece Chat modelini Foundry üzerinden yüklüyoruz
    chat_model = manager.catalog.get_model("phi-3.5-mini")
    chat_model.load()
    
    return chat_model.get_chat_client()

def get_embeddings(embedding_model, texts):
    """Verilen metin listesini yerel sentence-transformers ile vektörlere çevirir."""
    embeddings = embedding_model.encode(texts)
    return [emb.tolist() for emb in embeddings]

def find_relevant_documents(question_embedding, doc_embeddings, top_k=2):
    """Kullanıcının sorusuna en benzer dokümanların indekslerini döndürür."""
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(question_embedding, doc_emb)
        similarities.append((i, sim))
    
    # Benzerlik oranına göre büyükten küçüğe sırala
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # En yüksek skora sahip ilk 'top_k' dokümanın indeksini döndür
    return [idx for idx, sim in similarities[:top_k]]

def main():
    chat_client = setup_models()
    
    print("\nYerel Embedding Modeli (HuggingFace) Yükleniyor...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("\nDokümanlar vektörlere (embedding) dönüştürülüyor...")
    doc_embeddings = get_embeddings(embedding_model, DOCUMENTS)
    print("Doküman indeksleme tamamlandı!")
    
    print("\n" + "="*50)
    print(" RAG ÇEKİRDEĞİNE HOŞ GELDİNİZ (Zero Network) ")
    print("="*50)
    print("Çıkmak için 'q' veya 'exit' yazabilirsiniz.\n")
    
    # Halüsinasyonu önleyecek olan güçlü sistem kuralı
    system_prompt = (
        "Sen güvenilir, yerel ve gizlilik odaklı bir AI asistanısın. "
        "Sana sağlanan BAĞLAM (Context) metnini kullanarak kullanıcının sorusunu cevapla. "
        "Eğer sorunun cevabı BAĞLAM içerisinde YOKSA, sadece 'Bilmiyorum' de, kendi bilgini kullanarak uydurma yapma. "
        "Cevap verirken bilginin geldiği KAYNAK numarasını mutlaka belirt."
    )
    
    while True:
        try:
            question = input("Sorunuz: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if question.lower() in ['q', 'exit', 'çıkış']:
            break
            
        if not question.strip():
            continue
            
        # 1. Aşama: Retrieval (Soruyu vektöre çevir ve arama yap)
        q_emb = get_embeddings(embedding_model, [question])[0]
        relevant_indices = find_relevant_documents(q_emb, doc_embeddings, top_k=2)
        
        # 2. Aşama: Bağlamı Oluşturma
        context_parts = []
        for i in relevant_indices:
            context_parts.append(f"[KAYNAK {i+1}]: {DOCUMENTS[i]}")
        context = "\n".join(context_parts)
        
        print("\n[🔎 Bulunan Alakalı Dokümanlar]")
        print(context)
        print("-" * 50)
        
        # 3. Aşama: Generation (Modele soruyu ve bağlamı gönder)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"BAĞLAM:\n{context}\n\nSORU:\n{question}"}
        ]
        
        print("\nCevap Üretiliyor...\n")
        response = chat_client.complete_chat(messages)
        print("🤖 Asistan:", response.choices[0].message.content)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
