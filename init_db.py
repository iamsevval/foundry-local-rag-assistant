import sqlite3
import os

DB_NAME = 'rag_store.db'

def init_db():
    # Şema dosyasını oku
    with open('schema.sql', 'r') as f:
        schema = f.read()

    # Veritabanına bağlan (dosya yoksa oluşturur)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Şemayı çalıştır
    cursor.executescript(schema)
    print(f"{DB_NAME} oluşturuldu ve schema.sql uygulandı.")

    # Test verisi ekle
    cursor.execute('''
        INSERT INTO documents (source, content, embedding)
        VALUES (?, ?, ?)
    ''', ('test.txt', 'Bu bir test dokümanıdır.', '[0.1, 0.2, 0.3]'))
    conn.commit()
    print("Test verisi eklendi.")

    # Veriyi oku ve doğrula
    cursor.execute('SELECT * FROM documents')
    rows = cursor.fetchall()
    print("\n--- Veritabanındaki Kayıtlar ---")
    for row in rows:
        print(row)
    print("--------------------------------\n")

    conn.close()

if __name__ == '__main__':
    init_db()
