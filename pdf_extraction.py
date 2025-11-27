# import os
# import logging
# from PyPDF2 import PdfReader

# logger = logging.getLogger(__name__)


# def extract_text_from_papers(directory: str = "papers"):
#     extracted_data = {}

#     try:
#         if not os.path.exists(directory):
#             logger.error(f"Directory not found: {directory}")
#             return {}

#         logger.info(f"Scanning directory for PDF files: {directory}")

#         for file in os.listdir(directory):
#             if file.lower().endswith(".pdf"):
#                 file_path = os.path.join(directory, file)
#                 logger.info(f"Processing PDF: {file_path}")

#                 try:
#                     reader = PdfReader(file_path)

#                     text_content = []
#                     for i, page in enumerate(reader.pages):
#                         try:
#                             text = page.extract_text() or ""
#                             text_content.append(text)
#                         except Exception as e:
#                             logger.warning(f"Failed to extract text from page {i} in {file}: {e}")

#                     full_text = "\n".join(text_content)

#                     extracted_data[file] = full_text
#                     logger.info(f"Successfully extracted text from: {file}")

#                 except Exception as e:
#                     logger.error(f"Error reading PDF '{file}': {e}")

#         logger.info("Text extraction from all PDFs completed successfully.")
#         return extracted_data

#     except Exception as e:
#         logger.error(f"Unhandled error during PDF extraction: {e}")
#         raise
import os
import logging
from typing import List, Tuple, Dict
from PyPDF2 import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INFO] %(message)s"
)

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extracts text from a single PDF file.
    """
    try:
        logging.info(f"Extracting: {pdf_path}")

        reader = PdfReader(pdf_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text.strip()

    except Exception as e:
        logging.error(f"Failed to extract {pdf_path}: {e}")
        return ""


def extract_pair_from_folder(folder_path: str) -> Tuple[str, str, str]:
    folder_name = os.path.basename(folder_path)

    try:
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

        if len(pdf_files) != 2:
            return folder_name, "", f"Folder {folder_name} does not contain exactly 2 PDFs."

        # sort alphabetically to keep consistent order
        pdf_files.sort()

        q_paper = os.path.join(folder_path, pdf_files[0])
        ans_sheet = os.path.join(folder_path, pdf_files[1])

        q_text = extract_pdf_text(q_paper)
        a_text = extract_pdf_text(ans_sheet)

        combined = f"--- QUESTION PAPER ---\n{q_text}\n\n--- ANSWER SHEET ---\n{a_text}"

        return folder_name, combined, ""

    except Exception as e:
        return folder_name, "", str(e)


def extract_all_papers(papers_root: str) -> Dict[str, str]:
    results = {}

    logging.info(f"Scanning papers directory: {papers_root}")

    for folder in os.listdir(papers_root):
        folder_path = os.path.join(papers_root, folder)

        if not os.path.isdir(folder_path):
            continue

        folder_name, combined_text, err = extract_pair_from_folder(folder_path)

        if err:
            logging.error(err)
            continue

        results[folder_name] = combined_text
        logging.info(f"Extracted pair from folder: {folder_name}")

    return results
