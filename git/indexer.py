import sqlite3
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import git.document_converter as document_converter
import time
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.schema import NodeRelationship

embedding_model = "microsoft/harrier-oss-v1-0.6b"

print("All indexer imports done")

target_file = ""
file_name = os.path.basename(target_file)

db_path = "rag_database.db"
faiss_path = "my_personal_project.faiss"

# 1. Database Setup
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Updated schema to include 'path'
cursor.execute('''
    CREATE TABLE IF NOT EXISTS document_chunks (
        chunk_id INTEGER PRIMARY KEY,
        path TEXT,
        text TEXT,
        source_file TEXT
    )
''')

# 2. Check if the file is already in the database
cursor.execute("SELECT 1 FROM document_chunks WHERE source_file = ? LIMIT 1", (file_name,))
if cursor.fetchone():
    print(f"File '{file_name}' is already in the database. Doing nothing.")
    conn.close()
    exit(0)

# 3. Extract Document Chunks
print(f"Extracting hierarchical chunks for '{target_file}'...")
all_nodes = document_converter.process_document_for_rag(target_file)


if not all_nodes:
    print("No text found in document. Exiting.")
    conn.close()
    exit(0)

# Isolate leaf nodes for embedding
leaf_nodes = get_leaf_nodes(all_nodes)


# Establish a continuous integer sequence for SQLite ID assignments
cursor.execute("SELECT MAX(chunk_id) FROM document_chunks")
max_id = cursor.fetchone()[0]
start_id = (max_id + 1) if max_id is not None else 0

# Map LlamaIndex UUIDs to integer IDs
uuid_to_int = {node.node_id: start_id + i for i, node in enumerate(all_nodes)}
uuid_to_node = {node.node_id: node for node in all_nodes}

def get_lineage_path(node):
    """Traverse relationships to build the path string (e.g., '1/4/12')."""
    path = [uuid_to_int[node.node_id]]
    curr = node
    while NodeRelationship.PARENT in curr.relationships:
        parent_uuid = curr.relationships[NodeRelationship.PARENT].node_id
        if parent_uuid in uuid_to_int:
            path.insert(0, uuid_to_int[parent_uuid])
            curr = uuid_to_node[parent_uuid]
        else:
            break
    return "/".join(map(str, path))


# Create a set of leaf node UUIDs for fast O(1) lookup
leaf_node_ids = {node.node_id for node in leaf_nodes}

# 4. Populate SQLite Database
print("Populating SQLite database with hierarchy...")
for node in all_nodes:
    chunk_id = uuid_to_int[node.node_id]
    path_str = get_lineage_path(node)

    # NEW LOGIC: Only save the actual text if the node is a leaf
    text_to_save = node.text if node.node_id in leaf_node_ids else ""

    cursor.execute('''
        INSERT INTO document_chunks (chunk_id, path, text, source_file)
        VALUES (?, ?, ?, ?)
    ''', (chunk_id, path_str, text_to_save, file_name))

conn.commit()
# conn.close() # Leave this as it is in your script
conn.close()

# 5. Load or Initialize FAISS Index
if os.path.exists(faiss_path):
    print(f"Loading existing FAISS index from '{faiss_path}'...")
    index = faiss.read_index(faiss_path)
else:
    print("No existing FAISS index found. A new one will be created.")
    index = None

# 6. Generate Embeddings & Add ONLY Leaves to FAISS
print("Loading embedding model on GPU...")
model = SentenceTransformer(embedding_model, device="cuda")

# Extract only leaf texts and their specific integer IDs
leaf_texts = [node.text for node in leaf_nodes]
leaf_int_ids = [uuid_to_int[node.node_id] for node in leaf_nodes]

encode_start = time.time()
print(f"Encoding {len(leaf_texts)} leaf chunks...")
embeddings = model.encode(leaf_texts, normalize_embeddings=True)
encode_end = time.time()
print(f"{encode_end - encode_start:.2f} seconds to encode leaves.")

if index is None:
    dimension = embeddings.shape[1]
    # We must wrap IndexFlatIP in IndexIDMap to assign our custom database IDs
    base_index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(base_index)

# Add vectors explicitly tied to their SQLite leaf integer IDs
index.add_with_ids(embeddings, np.array(leaf_int_ids, dtype=np.int64))

print(f"Added {len(leaf_texts)} leaf vectors. Total vectors in FAISS index: {index.ntotal}.")

# 7. Save the Index
faiss.write_index(index, faiss_path)
print("Ingestion complete. Database and FAISS index updated and saved.")