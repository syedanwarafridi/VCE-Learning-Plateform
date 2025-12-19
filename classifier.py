import os
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from nomic import embed
import json

DB_CONFIG = {
    "host": "localhost",
    "database": "vce_learning_platform",
    "user": "postgres",
    "password": "postgres1234",
    "port": 5432
}

os.environ["NOMIC_API_KEY"] = "YOUR_API_KEY_HERE"


def get_embedding(text: str):
    resp = embed.text(
        texts=[text],
        model="nomic-embed-text-v1.5",
        task_type="search_query"
    )
    return np.array(resp["embeddings"][0], dtype=float)


def retrieve_similar(query: str, top_k: int = 3):
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    q_vec = get_embedding(query)

    sql = """
    SELECT 
        q.question_id, q.question_number, q.section, q.unit, q.aos, q.subtopic,
        q.skill_type, q.difficulty_level, q.question_text, q.answer_text,
        q.detailed_answer, q.page_number,
        e.exam_id, e.year, e.subject, e.unit AS exam_unit, e.exam_name,
        e.pdf_url, e.source
    FROM questions q
    JOIN exams e ON q.exam_id = e.exam_id
    ORDER BY q.embedding <-> %s
    LIMIT %s;
    """

    cur.execute(sql, (q_vec, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Return only the required fields
    result = []
    for r in rows:
        result.append({
            "subject": r[14],
            "unit": r[3],
            "area_of_study": r[4],
            "subtopic": r[5],
            "skill_type": r[6],
            "difficulty_level": r[7]
        })

    return result


if __name__ == "__main__":
    query = "Let f : R → R, f (x) = x(x − 2)2"
    matches = retrieve_similar(query, top_k=1)
    print(json.dumps(matches, indent=2, ensure_ascii=False))
