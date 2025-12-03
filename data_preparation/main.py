import os
import json
import logging
import time
from ai_model import load_grok, generate_text
from pdf_extraction import extract_text_from_folder, get_all_paper_folders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a precise PDF-to-JSON extractor. Your only job is to output **exactly** the requested JSON — nothing else.

=== ABSOLUTE RULES (NEVER VIOLATE ANY OF THESE) ===
1. Output **ONLY** valid JSON. Nothing before, after, or around it. No markdown, no ```json, no explanations, no reasoning, no notes.
2. If text is corrupted, broken, or OCR-garbled (e.g. "e3x", "xxe", "32323"), dont use it.
3. NEVER add "(1 mark)", "Method 1", "OR", "Note:", arrows, diagrams, or any commentary that is not part of the original question.
4. For question_text: extract ONLY the question stem as it appears in the PDF. Do NOT include marks.
6. Use \\n only when real line breaks exist inside question_text (very rare). Never add them otherwise.
6. Do NOT add any images, tables, or diagrams unless their exact URL appears in the PDF.
7. Metadata must be filled exactly as below — change ONLY what is in machine representable placeholders.
8. Questions, Answers, and Detailed Answered must be filled with valid data from pdf text.
9. Escape all other backslashes (\\) in text, including LaTeX, except for real line breaks \\n.
10. For `answer_text`, write **only the final answer** exactly as it should be, do NOT copy any corrupted symbols, OCR errors, or PDF annotations. Never include intermediate marks, steps, or matrix elements unless they are part of `detailed_answer`.
11. For `detailed_answer`, reconstruct the calculation logically based on the PDF text. Do NOT copy any corrupted text, extra numbers, brackets, or random symbols. Use real numbers and math expressions clearly, keeping \\n for line breaks only.
12. If the PDF text is unclear, estimate the answer logically but do not invent extra irrelevant symbols.

=== JSON STRUCTURE (fill exactly this, no extra fields, no trailing commas) ===
{{
"metadata": {{
    "scraped_at": "2025-11-22T06:45:00Z",
    "source": "VCAA Official Website",
}},
"exams": [
    {{
    "year": 2023,
    "subject": "Mathematical Methods",
    "unit": "Units 3 & 4",
    "exam": "Exam 1",
    "pdf_url": "https://www.vcaa.vic.edu.au/.../MM-Exam1-2023.pdf",
    "aos_breakdown": [
        {{
        "aos": "AOS 1: Functions and Graphs",
        "percentage": 40
        }}
    ],
    "questions": [
        {{
        "question_id": "MM-2023-E1-Q1",
        "question_number": 1,
        "section": "A",
        "unit": "Unit 3",
        "aos": "AOS 2: Algebra",
        "subtopic": "Quadratic Functions",
        "skill_type": "Procedural / Computation",
        "difficulty_level": "Easy",
        "question_text": "Let f(x) = 3x² − 5x + 2. Find f(3).",
        "answer_text": "f(3) = 14",
        "detailed_answer": "Step 1: Write down the function: f(x) = 3x² - 5x + 2\\nStep 2: Substitute x = 3 into the function: f(3) = 3(3)² - 5(3) + 2\\nStep 3: Calculate the square: (3)² = 9, so it becomes 3*9 - 5*3 + 2\\nStep 4: Perform the multiplications: 27 - 15 + 2\\nStep 5: Perform the addition and subtraction from left to right: 27 - 15 = 12, then 12 + 2 = 14\\nStep 6: State the final answer: Therefore, f(3) = 14",
        "subparts": [],
        "page_number": 3
        }}
    ]
    }}
]
}}

=== PDF CONTENT STARTS HERE ===
{combined_text}
=== PDF CONTENT ENDS HERE ===

Always output valid JSON:
- Strings must be enclosed in double quotes.
- Do not include raw OCR symbols or unescaped quotes.
- Only include printable characters.
- If text contains quotes, escape them properly.


Now output ONLY the final valid JSON. No other text allowed.
"""


def combine_pdf_texts(pdf_texts: dict) -> str:
    combined = []
    for filename, text in pdf_texts.items():
        combined.append(f"--- FILE: {filename} ---")
        combined.append(text)
        combined.append("")
    return "\n".join(combined)

def extract_json_from_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning("Attempting to extract JSON substring...")
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
        else:
            raise ValueError("No JSON object found in response")

def process_single_folder(folder_name: str, client, output_dir: str = "outputs"):
    folder_path = os.path.join("papers", folder_name)
    pdf_texts = extract_text_from_folder(folder_path)
    if not pdf_texts:
        logger.warning(f"No PDFs found in folder: {folder_name}")
        return False

    combined_text = combine_pdf_texts(pdf_texts)
    final_prompt = PROMPT_TEMPLATE.format(combined_text=combined_text)
    response = generate_text(client, final_prompt)
    if response.strip().startswith("json```") and response.strip().endswith("```"):
        response = response.strip()[7:-3].strip()
    print(response, "##############")
    
    if not response:
        logger.error(f"No response from model for folder {folder_name}")
        return False

    json_data = extract_json_from_response(response)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"folder_{folder_name}_output.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    logger.info(f"Saved JSON output: {save_path}")
    return True

def main():
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    client = load_grok(API_KEY)

    paper_folders = get_all_paper_folders()
    if not paper_folders:
        logger.error("No paper folders found.")
        return

    successful = 0
    for i, folder in enumerate(paper_folders):
        logger.info(f"=== Processing folder: {folder} ({i+1}/{len(paper_folders)}) ===")
        if process_single_folder(folder, client):
            successful += 1
            logger.info(f"✓ Folder processed: {folder}")
        else:
            logger.error(f"✗ Failed to process folder: {folder}")
        logger.info(f"=== Finished folder: {folder} ===\n")

        # -----------------------------
        # Wait 10 minutes before next request
        # -----------------------------
        if i < len(paper_folders) - 1:  # No need to wait after last folder
            logger.info("Waiting 1 seconds before next API request...")
            time.sleep(1)

    logger.info(f"Completed processing {successful}/{len(paper_folders)} folders.")

if __name__ == "__main__":
    main()
