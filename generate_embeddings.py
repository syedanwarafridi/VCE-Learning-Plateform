# generate_embeddings.py
import os
import psycopg2
from psycopg2.extras import execute_batch
from nomic import embed

# --- Provide your API key ---
os.environ["NOMIC_API_KEY"] = ""

# --- PostgreSQL config ---
DB_CONFIG = {
    "host": "localhost",
    "database": "vce_learning_platform",
    "user": "postgres",
    "password": "postgres1234",
    "port": 5432
}

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Fetch questions without embeddings
    cur.execute("SELECT question_id, question_text FROM questions WHERE embedding IS NULL;")
    rows = cur.fetchall()
    print(f"Found {len(rows)} questions to embed.")

    update_records = []
    for qid, qtext in rows:
        try:
            out = embed.text(
                texts=[qtext],
                model="nomic-embed-text-v1.5",
                task_type="search_document"
            )
            vector = out["embeddings"][0]
            update_records.append((vector, qid))
        except Exception as e:
            print(f"⚠️  Failed embedding for question {qid}: {e}")

    if update_records:
        update_sql = "UPDATE questions SET embedding = %s WHERE question_id = %s;"
        execute_batch(cur, update_sql, update_records)
        conn.commit()
        print("✅ Embeddings inserted into PostgreSQL!")
    else:
        print("ℹ️  No embeddings to insert.")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
