import sqlite3
# pyrefly: ignore [missing-import]
import sqlite_vec
import struct
from typing import List

DB_PATH = "rag_database.sqlite"

def serialize_f32(vector: List[float]) -> bytes:
    """Float listesini sqlite-vec'in anlayacağı binary (BLOB) formata çevirir."""
    return struct.pack("%sf" % len(vector), *vector)

def init_db():
    """Veritabanını ve sqlite-vec eklentisini başlatır."""
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    
    # Metadata tablosu
    db.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            content TEXT
        );
    """)
    
    # sqlite-vec Vektör tablosu (virtual table)
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[384]
        );
    """)
    
    # FTS5 Full Text Search tablosu (virtual table)
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_documents USING fts5(
            id UNINDEXED,
            source UNINDEXED,
            content,
            tokenize='porter'
        );
    """)
    
    db.commit()
    return db

def insert_chunk(source: str, content: str, embedding: List[float]):
    """Bir doküman parçasını hem Vektör hem FTS tablosuna kaydeder."""
    db = init_db()
    cursor = db.cursor()
    
    # 1. Metadatayı kaydet
    cursor.execute(
        "INSERT INTO documents (source, content) VALUES (?, ?)", 
        (source, content)
    )
    doc_id = cursor.lastrowid
    
    # 2. Vektörü kaydet
    cursor.execute(
        "INSERT INTO vec_documents (id, embedding) VALUES (?, ?)", 
        (doc_id, serialize_f32(embedding))
    )
    
    # 3. FTS5 (Kelime Araması) için kaydet
    cursor.execute(
        "INSERT INTO fts_documents (id, source, content) VALUES (?, ?, ?)", 
        (doc_id, source, content)
    )
    
    db.commit()
    db.close()
    return doc_id

def search_hybrid(query_text: str, query_embedding: List[float], top_k: int = 3) -> List[dict]:
    """
    Hem vektör (anlamsal) hem FTS5 (kelime) araması yapar,
    sonuçları RRF (Reciprocal Rank Fusion) ile birleştirir.
    """
    db = init_db()
    cursor = db.cursor()
    
    # 1. Vektör Araması (Anlamsal)
    query_blob = serialize_f32(query_embedding)
    cursor.execute("""
        SELECT id, vec_distance_cosine(embedding, ?) as distance
        FROM vec_documents
        WHERE embedding MATCH ? AND k = ?
    """, (query_blob, query_blob, top_k * 2))
    vec_results = cursor.fetchall()
    
    # 2. FTS5 Araması (Kelime tabanlı)
    safe_query = query_text.replace('"', '""').replace("'", "''")
    fts_results = []
    try:
        cursor.execute("""
            SELECT id, rank
            FROM fts_documents
            WHERE fts_documents MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (safe_query, top_k * 2))
        fts_results = cursor.fetchall()
    except sqlite3.OperationalError:
        pass # FTS MATCH bazen özel karakterlerde patlayabilir, yok sayıyoruz
    
    # 3. RRF (Reciprocal Rank Fusion) ile Birleştirme
    K = 60
    scores = {}
    
    for rank, (doc_id, distance) in enumerate(vec_results):
        scores[doc_id] = scores.get(doc_id, 0) + (1 / (K + rank))
        
    for rank, (doc_id, rank_score) in enumerate(fts_results):
        scores[doc_id] = scores.get(doc_id, 0) + (1 / (K + rank))
        
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
    
    results = []
    if sorted_ids:
        placeholders = ','.join('?' for _ in sorted_ids)
        order_cases = " ".join(f"WHEN {id_} THEN {i}" for i, id_ in enumerate(sorted_ids))
        cursor.execute(f"""
            SELECT id, source, content
            FROM documents
            WHERE id IN ({placeholders})
            ORDER BY CASE id {order_cases} END
        """, sorted_ids)
        
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "source": row[1],
                "content": row[2],
                "distance": scores[row[0]] # Arayüzde RRF skoru görünecek
            })
            
    db.close()
    return results

def clear_database():
    """Tüm veritabanı tablolarını temizler."""
    db = init_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM documents")
    cursor.execute("DELETE FROM vec_documents")
    cursor.execute("DELETE FROM fts_documents")
    db.commit()
    cursor.execute("VACUUM")
    db.commit()
    db.close()

def get_loaded_documents() -> List[str]:
    """Veritabanında bulunan eşsiz dosya isimlerini döndürür."""
    db = init_db()
    cursor = db.cursor()
    cursor.execute("SELECT DISTINCT source FROM documents")
    docs = [row[0] for row in cursor.fetchall()]
    db.close()
    return docs
