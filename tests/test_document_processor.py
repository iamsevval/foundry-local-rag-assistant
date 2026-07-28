import pytest
import os
import tempfile
from src.document_processor import semantic_chunking, process_and_chunk_file

def test_semantic_chunking():
    """Test that text is chunked properly with overlaps."""
    text = "This is a test sentence. " * 50  # Large text block with spaces
    chunks = semantic_chunking(text, max_chunk_size=500, overlap_size=50)
    
    assert len(chunks) > 1, "Text should be split into multiple chunks"
    assert len(chunks[0]) <= 550, "Chunk size should respect the limit"
    assert "test sentence" in chunks[0], "Chunk should contain the original text content"

def test_process_document_unsupported_file():
    """Test that unsupported files raise the correct RuntimeError."""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
        tmp.write(b"dummy data")
        tmp_path = tmp.name
        
    try:
        with pytest.raises(RuntimeError) as excinfo:
            process_and_chunk_file(tmp_path)
        assert "Desteklenmeyen dosya format" in str(excinfo.value) or "Dosya okuma hatas" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

def test_process_document_txt_file():
    """Test processing a simple TXT file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False, encoding='utf-8') as tmp:
        tmp.write("Bu bir test belgesidir.\nYapay zeka projelerinde test yazmak önemlidir.")
        tmp_path = tmp.name
        
    try:
        chunks = process_and_chunk_file(tmp_path)
        assert len(chunks) > 0, "TXT file should be successfully chunked"
        assert "test belgesidir" in chunks[0], "Content should be preserved"
    finally:
        os.remove(tmp_path)
