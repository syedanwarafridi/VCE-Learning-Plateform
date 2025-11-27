# import os
# import json
# import logging

# from ai_model import load_deepseek, generate_text
# from pdf_extraction import extract_text_from_papers

# # -----------------------------------------------------
# # Configure Logging
# # -----------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )
# logger = logging.getLogger(__name__)

# # -----------------------------------------------------
# # MAIN EXECUTION
# # -----------------------------------------------------
# def main():
#     # -------------------------------
#     # Load DeepSeek Model Once
#     # -------------------------------
#     logger.info("Initializing DeepSeek model...")
#     model, tokenizer = load_deepseek(model_type="7b", device="cuda")

#     # -------------------------------
#     # Extract PDF Text
#     # -------------------------------
#     pdf_texts = extract_text_from_papers(directory="papers")

#     if not pdf_texts:
#         logger.error("No PDFs found or extraction failed.")
#         return

#     # -------------------------------
#     # Custom Prompt
#     # -------------------------------
#     prompt_template = """
#     Read the below JSON schema. I will provide PDF files of Maths papers from VCAA/Haese/Insight. You need to extract the information from the PDFs in the below JSON schema format.
#     - Use "\n" in question_text and detailed_answer fields to represent line breaks
#     - Return ONLY the JSON output - no additional text, explanations, or markdown formatting
#     - The response should be pure JSON that can be directly parsed
    
#     Output structure:
#     {
#     "metadata": {
#         "scraped_at": "2025-11-22T06:45:00Z",
#         "source": "VCAA Official Website",
#         "schema_version": "1.2"
#     },
#     "exams": [
#         {
#         "year": 2023,
#         "subject": "Mathematical Methods",
#         "unit": "Units 3 & 4",
#         "exam": "Exam 1",
#         "pdf_url": "https://www.vcaa.vic.edu.au/.../MM-Exam1-2023.pdf",
#         "aos_breakdown": [
#             {
#             "aos": "AOS 1: Functions and Graphs",
#             "percentage": 40
#             },
#             {
#             "aos": "AOS 2: Algebra",
#             "percentage": 25
#             },
#             {
#             "aos": "AOS 3: Calculus",
#             "percentage": 35
#             }
#         ],
#         "questions": [
#             {
#             "question_id": "MM-2023-E1-Q1",
#             "question_number": 1,
#             "section": "Section A",
#             "unit": "Unit 3",
#             "aos": "AOS 2: Algebra",
#             "subtopic": "Quadratic Functions",
#             "skill_type": "Procedural / Computation",
#             "difficulty_level": "Easy",
#             "question_text": "Let f(x) = 3x² − 5x + 2. Find f(3).",
#             "answer_text": "f(3) = 14",
#             "detailed_answer": "Step 1: Write down the function: f(x) = 3x² - 5x + 2\nStep 2: Substitute x = 3 into the function: f(3) = 3(3)² - 5(3) + 2\nStep 3: Calculate the square: (3)² = 9, so it becomes 3*9 - 5*3 + 2\nStep 4: Perform the multiplications: 27 - 15 + 2\nStep 5: Perform the addition and subtraction from left to right: 27 - 15 = 12, then 12 + 2 = 14\nStep 6: State the final answer: Therefore, f(3) = 14",
#             "subparts": [],
#             "images": [],
#             "page_number": 3
#             }
#         ]
#         }
#     ]
#     }
    
#     PDF File: {pdf_texts}
#     """

#     # -------------------------------
#     # Process each PDF
#     # -------------------------------
#     output_dir = "outputs"
#     os.makedirs(output_dir, exist_ok=True)

#     for pdf_name, text in pdf_texts.items():
#         logger.info(f"Running model inference for: {pdf_name}")

#         # Final prompt for this PDF
#         final_prompt = prompt_template.format(content=text)

#         try:
#             # Generate model output
#             response = generate_text(model, tokenizer, final_prompt, max_tokens=500)

#             # Attempt to parse JSON from model output
#             try:
#                 json_data = json.loads(response)

#             except json.JSONDecodeError:
#                 logger.warning(f"Model output for {pdf_name} was not valid JSON.")
#                 logger.info("Attempting to extract JSON substring...")

#                 # Try to extract the JSON inside text
#                 try:
#                     start = response.index("{")
#                     end = response.rindex("}") + 1
#                     json_str = response[start:end]
#                     json_data = json.loads(json_str)
#                 except Exception as e:
#                     logger.error(f"Failed to recover JSON for {pdf_name}: {e}")
#                     continue

#             # Save JSON as file
#             json_filename = os.path.splitext(pdf_name)[0] + ".json"
#             save_path = os.path.join(output_dir, json_filename)

#             with open(save_path, "w", encoding="utf-8") as f:
#                 json.dump(json_data, f, indent=4, ensure_ascii=False)

#             logger.info(f"Saved JSON output: {save_path}")

#         except Exception as e:
#             logger.error(f"Error processing {pdf_name}: {e}")

#     logger.info("All PDFs processed.")


# # -----------------------------------------------------
# # Script Entry Point
# # -----------------------------------------------------
# if __name__ == "__main__":
#     main()


import os
import json
import logging
from pathlib import Path
from huggingface_hub import InferenceClient

from pdf_extraction import extract_all_papers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INFO] %(message)s"
)

# ---------------------------------------------------------
# Load your model once at startup
# ---------------------------------------------------------
def load_model(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    try:
        logging.info(f"Loading model: {model_name}")
        client = InferenceClient(model_name)
        logging.info("Model loaded successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise


# ---------------------------------------------------------
# Send text + prompt to model
# ---------------------------------------------------------
def generate_json_with_model(client, prompt: str, text: str) -> dict:
    try:
        full_input = f"{prompt}\n\nCONTENT:\n{text}"

        response = client.text_generation(
            full_input,
            max_new_tokens=2048,
            temperature=0.3,
        )

        # Sometimes models include extra text, ensure JSON extraction
        json_start = response.find("{")
        json_end = response.rfind("}")

        if json_start == -1 or json_end == -1:
            raise ValueError("Model output does not contain valid JSON.")

        json_str = response[json_start:json_end + 1]
        return json.loads(json_str)

    except Exception as e:
        logging.error(f"Model generation failed: {e}")
        return {}


# ---------------------------------------------------------
# Save JSON to file
# ---------------------------------------------------------
def save_json(output: dict, filename: str):
    Path("outputs").mkdir(exist_ok=True)

    output_path = os.path.join("outputs", f"{filename}.json")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        logging.info(f"Saved JSON: {output_path}")

    except Exception as e:
        logging.error(f"Failed to save JSON file {filename}: {e}")


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
def main():
    # Load model
    client = load_model()

    # Your prompt (insert your real prompt here)
    prompt = """
    Read the below JSON schema. I will provide PDF files of Maths papers from VCAA/Haese/Insight. You need to extract the information from the PDFs in the below JSON schema format.
    - Use "\n" in question_text and detailed_answer fields to represent line breaks
    - Return ONLY the JSON output - no additional text, explanations, or markdown formatting
    - The response should be pure JSON that can be directly parsed
    
    Output structure:
    {
    "metadata": {
        "scraped_at": "2025-11-22T06:45:00Z",
        "source": "VCAA Official Website",
        "schema_version": "1.2"
    },
    "exams": [
        {
        "year": 2023,
        "subject": "Mathematical Methods",
        "unit": "Units 3 & 4",
        "exam": "Exam 1",
        "pdf_url": "https://www.vcaa.vic.edu.au/.../MM-Exam1-2023.pdf",
        "aos_breakdown": [
            {
            "aos": "AOS 1: Functions and Graphs",
            "percentage": 40
            },
            {
            "aos": "AOS 2: Algebra",
            "percentage": 25
            },
            {
            "aos": "AOS 3: Calculus",
            "percentage": 35
            }
        ],
        "questions": [
            {
            "question_id": "MM-2023-E1-Q1",
            "question_number": 1,
            "section": "Section A",
            "unit": "Unit 3",
            "aos": "AOS 2: Algebra",
            "subtopic": "Quadratic Functions",
            "skill_type": "Procedural / Computation",
            "difficulty_level": "Easy",
            "question_text": "Let f(x) = 3x² − 5x + 2. Find f(3).",
            "answer_text": "f(3) = 14",
            "detailed_answer": "Step 1: Write down the function: f(x) = 3x² - 5x + 2\nStep 2: Substitute x = 3 into the function: f(3) = 3(3)² - 5(3) + 2\nStep 3: Calculate the square: (3)² = 9, so it becomes 3*9 - 5*3 + 2\nStep 4: Perform the multiplications: 27 - 15 + 2\nStep 5: Perform the addition and subtraction from left to right: 27 - 15 = 12, then 12 + 2 = 14\nStep 6: State the final answer: Therefore, f(3) = 14",
            "subparts": [],
            "images": [],
            "page_number": 3
            }
        ]
        }
    ]
    }
    
    PDF File: {pdf_texts}
    """

    # Extract all papers (each folder -> combined question + answer text)
    papers_dir = "papers"
    folder_texts = extract_all_papers(papers_dir)

    for folder_name, combined_text in folder_texts.items():
        logging.info(f"Processing folder: {folder_name}")

        output_json = generate_json_with_model(
            client=client,
            prompt=prompt,
            text=combined_text
        )

        if output_json:
            save_json(output_json, folder_name)
        else:
            logging.error(f"Skipping saving for {folder_name} due to empty output.")


if __name__ == "__main__":
    main()
