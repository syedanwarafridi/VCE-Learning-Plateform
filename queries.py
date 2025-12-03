import psycopg2, json

def get_question_by_index(index=5):

    conn = psycopg2.connect(
        host="localhost",
        database="vce_learning_platform",
        user="postgres",
        password="postgres1234",
        port="5432"
    )
    cursor = conn.cursor()

    # Step 1 — get question_id at index 5
    cursor.execute("""
        SELECT question_id 
        FROM questions 
        ORDER BY question_id 
        OFFSET %s LIMIT 1;
    """, (index,))

    row = cursor.fetchone()
    if not row:
        print(f"No question exists at index {index}")
        return

    question_id = row[0]
    print(f"📌 Selected Question ID at index {index}: {question_id}")

    # Step 2 — fetch full linked details
    cursor.execute("""
        SELECT  
            q.question_id, q.question_number, q.section, q.unit, q.aos, q.subtopic,
            q.skill_type, q.difficulty_level, q.question_text, q.answer_text, 
            q.detailed_answer, q.page_number,

            e.exam_id, e.year, e.subject, e.unit AS exam_unit, e.exam_name,
            e.pdf_url, e.source, e.scraped_at,

            ab.aos_name, ab.percentage,

            sp.subpart_id, sp.subpart_letter, sp.subpart_text, 
            sp.subpart_answer, sp.subpart_detailed_answer

        FROM questions q
        JOIN exams e ON q.exam_id = e.exam_id
        LEFT JOIN aos_breakdown ab ON ab.exam_id = e.exam_id
        LEFT JOIN question_subparts sp ON sp.question_id = q.question_id
        WHERE q.question_id = %s
        ORDER BY sp.subpart_letter;
    """, (question_id,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Transform to structured JSON
    result = {
        "question_id": rows[0][0],
        "question_number": rows[0][1],
        "section": rows[0][2],
        "unit": rows[0][3],
        "aos": rows[0][4],
        "subtopic": rows[0][5],
        "skill_type": rows[0][6],
        "difficulty_level": rows[0][7],
        "question_text": rows[0][8],
        "answer_text": rows[0][9],
        "detailed_answer": rows[0][10],
        "page_number": rows[0][11],
        "exam": {
            "exam_id": rows[0][12],
            "year": rows[0][13],
            "subject": rows[0][14],
            "unit": rows[0][15],
            "exam_name": rows[0][16],
            "pdf_url": rows[0][17],
            "source": rows[0][18],
            "scraped_at": rows[0][19]
        },
        "aos_breakdown": [],
        "subparts": []
    }

    for r in rows:
        if r[20]: result["aos_breakdown"].append({"aos_name": r[20], "percentage": r[21]})
        if r[22]: result["subparts"].append({
            "subpart_id": r[22],
            "subpart_letter": r[23],
            "subpart_text": r[24],
            "subpart_answer": r[25],
            "subpart_detailed_answer": r[26]
        })

    print(json.dumps(result, indent=4, default=str))
    return result


# Run it
get_question_by_index(5)
