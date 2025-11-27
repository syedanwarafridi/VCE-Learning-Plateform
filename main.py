import os
import json
import logging

from ai_model import load_deepseek, generate_text
from pdf_extraction import extract_text_from_folder, get_all_paper_folders

# -----------------------------------------------------
# Configure Logging
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asasctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Custom Prompt Template
# -----------------------------------------------------
PROMPT_TEMPLATE = """
Read the below JSON schema. I will provide PDF files of Maths papers from VCAA/Haese/Insight. You need to extract the information from the PDFs in the below JSON schema format.
- Use "\\n" in question_text and detailed_answer fields to represent line breaks
- Return ONLY the JSON output - no additional text, explanations, or markdown formatting
- The response should be pure JSON that can be directly parsed

Output structure:
{{
"metadata": {{
    "scraped_at": "2025-11-22T06:45:00Z",
    "source": "VCAA Official Website",
    "schema_version": "1.2"
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
        }},
        {{
        "aos": "AOS 2: Algebra",
        "percentage": 25
        }},
        {{
        "aos": "AOS 3: Calculus",
        "percentage": 35
        }}
    ],
    "questions": [
        {{
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
        "detailed_answer": "Step 1: Write down the function: f(x) = 3x² - 5x + 2\\nStep 2: Substitute x = 3 into the function: f(3) = 3(3)² - 5(3) + 2\\nStep 3: Calculate the square: (3)² = 9, so it becomes 3*9 - 5*3 + 2\\nStep 4: Perform the multiplications: 27 - 15 + 2\\nStep 5: Perform the addition and subtraction from left to right: 27 - 15 = 12, then 12 + 2 = 14\\nStep 6: State the final answer: Therefore, f(3) = 14",
        "subparts": [],
        "images": [],
        "page_number": 3
        }}
    ]
    }}
]
}}

PDF Content from folder:
{combined_text}
"""


def combine_pdf_texts(pdf_texts: dict) -> str:
    """
    Combine text from multiple PDFs into a single string with clear separation
    """
    combined = []
    for filename, text in pdf_texts.items():
        combined.append(f"--- FILE: {filename} ---")
        combined.append(text)
        combined.append("")  # Empty line between files
    
    return "\n".join(combined)


def extract_json_from_response(response: str) -> dict:
    """
    Extract JSON from model response, handling cases where there's extra text
    """
    try:
        # First try direct JSON parsing
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning("Model output was not valid JSON. Attempting to extract JSON substring...")
        
        try:
            # Try to find JSON object in the response
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")
                
        except Exception as e:
            logger.error(f"Failed to extract JSON from response: {e}")
            logger.debug(f"Problematic response: {response}")
            raise


def process_single_folder(folder_name: str, model, tokenizer, output_dir: str = "outputs"):
    """
    Process a single folder: extract PDFs, combine text, run model, save JSON
    """
    try:
        folder_path = os.path.join("papers", folder_name)
        logger.info(f"Processing folder: {folder_name}")
        
        # Extract text from all PDFs in the folder
        pdf_texts = extract_text_from_folder(folder_path)
        
        if not pdf_texts:
            logger.warning(f"No PDFs found in folder: {folder_name}")
            return False
        
        logger.info(f"Found {len(pdf_texts)} PDF files in folder {folder_name}")
        
        # Combine text from all PDFs
        combined_text = combine_pdf_texts(pdf_texts)
        logger.info(f"Combined text length: {len(combined_text)} characters")
        
        # Prepare final prompt
        final_prompt = PROMPT_TEMPLATE.format(combined_text=combined_text)
        
        # Generate model response
        logger.info(f"Running model inference for folder: {folder_name}")
        response = generate_text(model, tokenizer, final_prompt, max_tokens=2000)
        
        # Extract JSON from response
        json_data = extract_json_from_response(response)
        
        # Save JSON file
        json_filename = f"folder_{folder_name}_output.json"
        save_path = os.path.join(output_dir, json_filename)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Successfully saved JSON output: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing folder {folder_name}: {e}")
        return False


# -----------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------
def main():
    # -------------------------------
    # Load DeepSeek Model Once
    # -------------------------------
    logger.info("Initializing DeepSeek model...")
    model, tokenizer = load_deepseek(model_type="7b", device="cuda")
    
    # -------------------------------
    # Get all paper folders
    # -------------------------------
    paper_folders = get_all_paper_folders()
    
    if not paper_folders:
        logger.error("No paper folders found.")
        return
    
    logger.info(f"Found {len(paper_folders)} folders to process")
    
    # -------------------------------
    # Create output directory
    # -------------------------------
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------
    # Process each folder sequentially
    # -------------------------------
    successful_processing = 0
    
    for folder_name in paper_folders:
        logger.info(f"=== Starting processing for folder: {folder_name} ===")
        
        success = process_single_folder(folder_name, model, tokenizer, output_dir)
        
        if success:
            successful_processing += 1
            logger.info(f"✓ Completed processing folder: {folder_name}")
        else:
            logger.error(f"✗ Failed to process folder: {folder_name}")
        
        logger.info(f"=== Finished processing folder: {folder_name} ===\n")
    
    # -------------------------------
    # Summary
    # -------------------------------
    logger.info(f"Processing completed. Successfully processed {successful_processing}/{len(paper_folders)} folders")


# -----------------------------------------------------
# Script Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    main()