import pytest
import sqlite3
import os
import tempfile
from src import vector_db

@pytest.fixture
def temp_db():
    """Create a temporary test database."""
    # We will override the DEFAULT_DB_PATH in vector_db for testing
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    
    original_path = vector_db.DB_PATH
    vector_db.DB_PATH = path
    
    yield path
    
    # Teardown
    vector_db.DB_PATH = original_path
    if os.path.exists(path):
        os.remove(path)

def test_init_db(temp_db):
    """Test database initialization and virtual tables creation."""
    vector_db.init_db()
    
    # Check if tables were created
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert "vec_documents" in tables, "vec_documents table should be created"
    assert "fts_documents" in tables, "fts_documents table should be created"
    assert "graph_edges" in tables, "graph_edges table should be created"
    
    conn.close()

def test_store_and_search_vector(temp_db):
    """Test basic vector storage without full hybrid setup."""
    vector_db.init_db()
    
    dummy_embedding = [0.1] * 384
    
    # Store dummy data
    conn = sqlite3.connect(temp_db)
    import sqlite_vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    cursor = conn.cursor()
    
    try:
        # Insert FTS
        cursor.execute("INSERT INTO fts_documents(source, content) VALUES (?, ?)", 
                       ("test.txt", "This is a test document about AI."))
        rowid = cursor.lastrowid
        
        # Insert VEC
        import struct
        emb_bytes = struct.pack(f"{len(dummy_embedding)}f", *dummy_embedding)
        cursor.execute("INSERT INTO vec_documents(rowid, embedding) VALUES (?, ?)", 
                       (rowid, emb_bytes))
        conn.commit()
        
        # Verify it exists
        cursor.execute("SELECT rowid FROM fts_documents WHERE content MATCH 'AI'")
        res = cursor.fetchall()
        assert len(res) == 1, "Should find the inserted document via FTS"
    finally:
        conn.close()
