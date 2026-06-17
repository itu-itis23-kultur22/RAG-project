import sqlite3
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Define the models
QUERY_MODEL = "microsoft/harrier-oss-v1-0.6b"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def extract_lineage_and_scores(faiss_indices, faiss_distances, cursor):
    """
    Extracts the hierarchical lineage (top and mid-level nodes) for the retrieved leaf chunks
    and calculates aggregated scores for parent nodes based on their children's FAISS distances.
    """

    # 1. Create the placeholder string: "?, ?, ?" to prevent SQL injection in the IN clause
    placeholders = ', '.join(['?'] * len(faiss_indices))

    # 2. Construct the query to fetch the hierarchical paths of the leaf chunks retrieved by FAISS
    query = f"SELECT path FROM document_chunks WHERE chunk_id IN ({placeholders})"

    # 3. Execute and fetch results (returns paths formatted like 'root_id/mid_id/leaf_id')
    cursor.execute(query, faiss_indices)
    results = cursor.fetchall()

    #print(results)

    top_levels = []

    # 4. Parse the retrieved paths to isolate unique top-level (root) and mid-level (parent) nodes
    for row in results:
        top_level = row[0].split("/")[0]
        if top_level not in top_levels:
            top_levels.append(top_level)

    # for i, faiss_index_enu in enumerate(faiss_indices):
    #    print(faiss_index_enu, faiss_distances[i])

    #print(faiss_indices)

    # 5. Find all descendant nodes that share the same top-level roots (the whole tree branch)
    query_parts = []
    params = []

    for tl in top_levels:
        query_parts.append("path LIKE ?")
        params.append(f"{tl}/%")  # The % acts as a SQL wildcard for any child branches

    # Join the conditions with OR to fetch all related lineage paths in a single query
    query2 = f"SELECT path FROM document_chunks WHERE {' OR '.join(query_parts)}"

    cursor.execute(query2, params)
    all_descendants = cursor.fetchall()

    #print(all_descendants)

    only_des = {}

    for row in all_descendants:
        row_nodes = row[0].split("/")
        for row_node in row_nodes:
            if row_node not in only_des:
                only_des[row_node] = []

            else:
                # if row_node is not leaf
                if row_nodes.index(row_node) != len(row_nodes) - 1:
                    next_node = row_nodes[row_nodes.index(row_node) + 1]
                    # if the child is not already added
                    if next_node not in only_des[row_node]:
                        only_des[row_node].append(next_node)

    # print("only_des")
    # print(only_des)
    min_score = min(faiss_distances)

    def find_score(search_node, search_dict, constructor_dict):

        if search_node in search_dict:
            return search_dict[search_node]

        # if node is a leaf
        if not constructor_dict[search_node]:
            # if the leaf is one of the faiss vectors
            if int(search_node) in faiss_indices:
                return faiss_distances[faiss_indices.index(int(search_node))]
            # if not, its score should be at most the smallest score amongst vectors
            else:
                return min_score
        # if not
        else:
            search_score = 0
            for child_search in constructor_dict[search_node]:
                search_score += find_score(child_search, search_dict, constructor_dict)

            return search_score / len(constructor_dict[search_node])

    des_scores = {}

    for score_node in only_des:
        des_scores[score_node] = find_score(score_node, des_scores, only_des)

    # ==========================================
    # --- START OF NEW BOTTOM-UP LOGIC ---
    # ==========================================

    # 1. Map each node to its depth in the tree structure
    node_depths = {}
    for row in all_descendants:
        row_nodes = row[0].split("/")
        for depth, node in enumerate(row_nodes):
            # A node might appear in multiple paths, keep its maximum depth
            if node not in node_depths or depth > node_depths[node]:
                node_depths[node] = depth

    deleted_nodes = []
    threshold_scores = des_scores.copy()

    # 2. Filter out leaf nodes and sort the remaining parents Bottom-Up

    # Step A: Isolate only the parent nodes (nodes that actually have children)
    parent_nodes = []
    for node_id in only_des:
        if only_des[node_id]:  # If the list of children is not empty
            parent_nodes.append(node_id)

    # Step B: Define a simple helper function for the sort to use
    def get_node_depth(n_id):
        return node_depths.get(n_id, 0)

    # Step C: Sort the parents by their depth descending (deepest parents first)
    bottom_up_nodes = sorted(parent_nodes, key=get_node_depth, reverse=True)


    # 3. Apply the thresholding logic starting from the lowest parents
    for node in bottom_up_nodes:
        if node not in deleted_nodes:
            score = des_scores[node]

            # find maximum score among child nodes
            max_score = 0
            for node_child in only_des[node]:
                child_score = des_scores[node_child]
                if max_score < child_score:
                    max_score = child_score

            # If a parent node's aggregated score is at least 96.5% as good as its best child's score,
            # the parent is kept and the children are deleted
            if score < max_score * 0.965: # this value is not necessarily optimal, just a starting point for experimentation
                deleted_nodes.append(node)
                if node in threshold_scores:
                    del threshold_scores[node]
            else:
                for node_child in only_des[node]:
                    if node_child not in deleted_nodes:
                        deleted_nodes.append(node_child)
                        if node_child in threshold_scores:
                            del threshold_scores[node_child]

    # ==========================================
    # --- END OF NEW BOTTOM-UP LOGIC ---
    # ==========================================

    no_min_th_scores = threshold_scores.copy()

    for node in threshold_scores:
        if threshold_scores[node] == min_score:
            del no_min_th_scores[node]

    #print(no_min_th_scores)

    return list(no_min_th_scores.keys()), no_min_th_scores


def search_rag(query: str, index, bi_encoder, cross_encoder, cursor,
               initial_top_k: int = 30, lineage_top_k: int = 15, final_top_k: int = 3):
    # 1. Format and encode the query using the Bi-Encoder
    formatted_query = f"Instruct: Retrieve semantically similar text\nQuery: {query}"
    #print(f"Encoding the query: '{query}'")
    query_embedding = bi_encoder.encode([formatted_query], normalize_embeddings=True)

    # 2. Perform the initial similarity search (FAISS)
    #print(f"Retrieving the top {initial_top_k} results from FAISS...")
    distances, indices = index.search(query_embedding, initial_top_k)

    # Clean up FAISS outputs
    faiss_indices = [int(idx) for idx in indices[0] if idx != -1]
    faiss_distances = [float(dist) for i, dist in enumerate(distances[0]) if indices[0][i] != -1]

    if not faiss_indices:
        print("No results found in FAISS.")
        return

    # 3. Apply hierarchical tree logic to get lineage nodes and their aggregated scores
    #print("Aggregating scores across tree lineage...")
    lineage_ids, aggregated_scores = extract_lineage_and_scores(faiss_indices, faiss_distances, cursor)

    if not lineage_ids:
        print("No valid lineage nodes derived from the retrieved leaves.")
        return

    #print(f"Filtering {len(lineage_ids)} lineage nodes down to top {lineage_top_k}...")

    # Sort the node IDs based on their aggregated score in descending order
    sorted_lineage_ids = sorted(lineage_ids, key=lambda node: aggregated_scores.get(node, 0.0), reverse=True)

    #for lineage_id in sorted_lineage_ids:
    #    print(lineage_id, aggregated_scores[lineage_id])

    # Slice to keep only the top K nodes
    top_lineage_ids = sorted_lineage_ids[:lineage_top_k]

    # 4. Connect to SQLite to retrieve text specifically for the lineage nodes
    retrieved_chunks = []

    for node_id_str in top_lineage_ids:
        # node_id_str is a string because of the path split logic, cast for SQL safety
        node_id = str(node_id_str)

        # Fetch the baseline data for the winning node
        cursor.execute("SELECT path, source_file, text FROM document_chunks WHERE chunk_id = ?", (node_id,))
        node_data = cursor.fetchone()

        if not node_data:
            continue

        path, source_file, stored_text = node_data

        # If stored_text is empty, this is a parent node. We must reconstruct its text from its leaves.
        if not stored_text.strip():

            # Use the EXACT path we already retrieved, plus a trailing wildcard.
            # E.g., if path is '1/4', we search for exact match '1/4' OR anything starting with '1/4/'
            reconstruction_query = """
                        SELECT text FROM document_chunks 
                        WHERE (path = ? OR path LIKE ?)
                        AND text != '' 
                        ORDER BY chunk_id ASC
                    """

            # The exact path, and the path acting as a strict prefix for children
            cursor.execute(
                reconstruction_query,
                (path, f"{path}/%")
            )
            leaf_results = cursor.fetchall()

            # Stitch the leaf texts back together chronologically
            actual_text = " ".join([row[0] for row in leaf_results])
        else:
            # It is already a leaf node, use the text directly
            actual_text = stored_text

        retrieved_chunks.append({
            "chunk_id": node_id,
            "path": path,
            "text": actual_text,
            "source_file": source_file,
            "tree_score": aggregated_scores.get(node_id_str, 0.0)
        })

    # 5. Rerank the retrieved lineage chunks using the Cross-Encoder
    #print(f"Reranking {len(retrieved_chunks)} lineage results...")

    # Format inputs for the Cross-Encoder: [[query, text1], [query, text2], ...]
    cross_inp = [[query, chunk["text"]] for chunk in retrieved_chunks]

    # Predict scores
    cross_scores = cross_encoder.predict(cross_inp)

    # Append scores to the dictionary and sort descending by cross_score
    for i, score in enumerate(cross_scores):
        retrieved_chunks[i]["cross_score"] = float(score)

    reranked_chunks = sorted(retrieved_chunks, key=lambda x: x["cross_score"], reverse=True)

    # 6. Output the final top results
    print("\n" + "=" * 60)
    print("🎯 LINEAGE RETRIEVAL & RERANKING RESULTS")
    print("=" * 60)

    print("Summary of reranked chunks:", [chunk["path"] for chunk in retrieved_chunks])
    for i in range(min(final_top_k, len(reranked_chunks))):
        chunk = reranked_chunks[i]
        print(f"\nRank: {i + 1} | Chunk ID: {chunk['chunk_id']} | Path: {chunk['path']}")
        print(f"Source: {chunk['source_file']}")
        print(f"Aggregated Tree Score: {chunk['tree_score']:.4f} | Cross-Encoder Score: {chunk['cross_score']:.4f}")
        print("-" * 60)
        print(chunk["text"])
        print("-" * 60)

    # ADD THIS TO THE END OF YOUR search_rag FUNCTION
    return reranked_chunks[:final_top_k]


if __name__ == "__main__":
    print("Initializing system and loading models into memory...")

    try:
        print("Loading FAISS index...")
        faiss_index = faiss.read_index("my_personal_project.faiss")
    except RuntimeError:
        print("Error: Could not find 'my_personal_project.faiss'. Run embed.py first.")
        exit(1)

    print("Loading embedding model (Bi-Encoder) on CPU...")
    # GPU for now since testing does not require the main LLM to be loaded
    bi_model = SentenceTransformer(QUERY_MODEL, device="cuda")

    print("Loading reranking model (Cross-Encoder) on CPU...")
    # GPU for now since testing does not require the main LLM to be loaded
    cross_model = CrossEncoder(RERANKER_MODEL, device="cuda", max_length=1024, trust_remote_code=True)

    print("Connecting to SQLite database...")
    db_conn = sqlite3.connect("rag_database.db")
    db_cursor = db_conn.cursor()

    print("\n✅ System ready!")

    while True:
        user_input = input("\nEnter your question (or 'q' to quit): ").strip()

        if user_input.lower() == 'q':
            print("Cleaning up and exiting...")
            db_conn.close()
            break


        if user_input:
            search_rag(
                query=user_input,
                index=faiss_index,
                bi_encoder=bi_model,
                cross_encoder=cross_model,
                cursor=db_cursor,
                initial_top_k=40,
                lineage_top_k=15,
                final_top_k=5
            )