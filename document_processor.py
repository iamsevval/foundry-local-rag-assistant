# pyrefly: ignore [missing-import]
import os
# pyrefly: ignore [missing-import]
import pypdf
# pyrefly: ignore [missing-import]
import docx
from typing import List

def extract_text_from_docx(file_path: str) -> str:
    """DOCX dosyasından tüm metni çıkarır."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text

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

def semantic_chunking(text: str, max_chunk_size: int = 1500, overlap_size: int = 300) -> List[str]:
    """
    Metni cümle sınırlarına (noktalara) dikkat ederek parçalara böler.
    'overlap' (örtüşme) mantığı ile parçaların son kısımlarını bir sonraki parçanın başına ekleyerek bağlam kopukluğunu önler.
    """
    # Basit bir cümle bölücü (Nokta + boşluk)
    sentences = [s.strip() + "." for s in text.replace('\n', ' ').split('. ') if s.strip()]
    
    chunks = []
    current_chunk = ""
    overlap_buffer = "" # Son eklenen cümleleri tutacağımız tampon
    
    for sentence in sentences:
        # Eğer mevcut chunk'a bu cümleyi eklersek limiti aşacak mıyız?
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Yeni chunk'a başlarken overlap (örtüşme) tamponunu en başa koyuyoruz
            current_chunk = overlap_buffer + sentence + " "
        else:
            current_chunk += sentence + " "
            
        # Overlap buffer'ı güncelle (sadece son overlap_size karaktere sığan cümleleri tut)
        overlap_buffer += sentence + " "
        while len(overlap_buffer) > overlap_size and ". " in overlap_buffer:
            overlap_buffer = overlap_buffer.split(". ", 1)[1]
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def process_and_chunk_file(file_path: str) -> List[str]:
    """Dosyayı okur ve parçalara (chunks) böler."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")
        
    return semantic_chunking(text)
