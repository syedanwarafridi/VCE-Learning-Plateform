import os
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from nomic import embed  # assuming nomic embed.text works

DB_CONFIG = {
    "host": "localhost",
    "database": "vce_learning_platform",
    "user": "postgres",
    "password": "postgres",
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
        q.question_id, q.question_number, q.section, q.unit, q.aos, q.subtopic, q.skill_type, q.difficulty_level,
        q.question_text, q.answer_text, q.detailed_answer, q.page_number,
        e.exam_id, e.year, e.subject, e.unit AS exam_unit, e.exam_name, e.pdf_url, e.source,
        (
            SELECT json_agg(json_build_object('aos_name', a.aos_name, 'percentage', a.percentage))
            FROM aos_breakdown a
            WHERE a.exam_id = e.exam_id
        ) AS aos_breakdown,
        (
            SELECT json_agg(json_build_object(
                'subpart_letter', sp.subpart_letter,
                'subpart_text', sp.subpart_text,
                'subpart_answer', sp.subpart_answer,
                'subpart_detailed_answer', sp.subpart_detailed_answer
            ))
            FROM question_subparts sp
            WHERE sp.question_id = q.question_id
        ) AS subparts
    FROM questions q
    JOIN exams e ON q.exam_id = e.exam_id
    ORDER BY q.embedding <-> %s
    LIMIT %s;
    """

    cur.execute(sql, (q_vec, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "question_id": r[0],
            "question_number": r[1],
            "section": r[2],
            "unit": r[3],
            "aos": r[4],
            "subtopic": r[5],
            "skill_type": r[6],
            "difficulty_level": r[7],
            "question_text": r[8],
            "answer_text": r[9],
            "detailed_answer": r[10],
            "page_number": r[11],
            "exam": {
                "exam_id": r[12],
                "year": r[13],
                "subject": r[14],
                "unit": r[15],
                "exam_name": r[16],
                "pdf_url": r[17],
                "source": r[18]
            },
            "aos_breakdown": r[19] or [],
            "subparts": r[20] or []
        })

    return results

if __name__ == "__main__":
    query = "Let f : R → R, f (x) = x(x − 2)2"
    matches = retrieve_similar(query, top_k=3)
    import json
    print(json.dumps(matches, indent=2, ensure_ascii=False))
