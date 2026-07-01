# pyrefly: ignore [missing-import]
import os
import pypdf
from typing import List

def extract_text_from_pdf(file_path: str) -> str:
    """PDF dosyasından tüm metni çıkarır."""
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def semantic_chunking(text: str, max_chunk_size: int = 1500) -> List[str]:
    """
    Metni cümle sınırlarına (noktalara) dikkat ederek parçalara böler.
    Körlemesine karakter sayısına göre bölmek yerine bağlamı korur.
    """
    # Basit bir cümle bölücü (Nokta + boşluk)
    sentences = [s.strip() + "." for s in text.replace('\n', ' ').split('. ') if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Eğer mevcut chunk'a bu cümleyi eklersek limiti aşacak mıyız?
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Yeni chunk'a başlarken overlap (örtüşme) için önceki cümlenin son kısmını da ekleyebiliriz
            # Şimdilik sadece yeni cümleden başlatıyoruz.
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def process_and_chunk_file(file_path: str) -> List[str]:
    """Dosyayı okur ve parçalara (chunks) böler."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")
        
    return semantic_chunking(text)
