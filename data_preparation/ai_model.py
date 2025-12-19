# import logging
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch

# # -----------------------------------------------------
# # Configure Logging
# # -----------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )

# logger = logging.getLogger(__name__)


# # -----------------------------------------------------
# # Load Tokenizer
# # -----------------------------------------------------
# def load_tokenizer(model_name: str):
#     try:
#         logger.info(f"Loading tokenizer for model: {model_name}")
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             trust_remote_code=True
#         )
#         logger.info("Tokenizer loaded successfully.")
#         return tokenizer

#     except Exception as e:
#         logger.error(f"Failed to load tokenizer: {e}")
#         raise


# # -----------------------------------------------------
# # Load Model in 4-bit Quantization (BitsAndBytes)
# # -----------------------------------------------------
# def load_model(model_name: str, device: str = "cuda"):
#     try:
#         logger.info(f"Loading model (4-bit quantized): {model_name}")

#         quant_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#         )

#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             quantization_config=quant_config,
#             device_map="auto",
#             trust_remote_code=True
#         )

#         logger.info("Model loaded successfully in 4-bit.")
#         return model

#     except Exception as e:
#         logger.error(f"Failed to load quantized model: {e}")
#         raise


# # -----------------------------------------------------
# # Text Generation (with safe long-input handling)
# # -----------------------------------------------------
# def generate_text(model, tokenizer, prompt: str, max_tokens: int = 200):
#     try:
#         logger.info("Tokenizing input...")

#         inputs = tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,         # Prevent OOM
#             max_length=4096          # Adjust as needed
#         ).to(model.device)

#         logger.info("Generating output...")
#         output = model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True
#         )

#         logger.info("Text generation completed.")
#         return tokenizer.decode(output[0], skip_special_tokens=True)

#     except Exception as e:
#         logger.error(f"Error during text generation: {e}")
#         raise


# # -----------------------------------------------------
# # Wrapper for 4B or 7B
# # -----------------------------------------------------
# def load_deepseek(model_type: str = "fast", device: str = "cuda"):
#     try:
#         model_map = {
#             "fast": "meta-llama/Llama-3.2-3B-Instruct",
#             "7b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
#             "4b": "Qwen/Qwen3-4B-Instruct-2507"
#         }

#         if model_type.lower() not in model_map:
#             raise ValueError("model_type must be '4b' or '7b'")

#         model_name = model_map[model_type.lower()]
#         logger.info(f"Selected model: {model_type.upper()} ({model_name})")

#         tokenizer = load_tokenizer(model_name)
#         model = load_model(model_name, device=device)

#         logger.info("Model + tokenizer loaded successfully.")
#         return model, tokenizer

#     except Exception as e:
#         logger.error(f"Failed to initialize DeepSeek/Qwen model: {e}")
#         raise


# # -----------------------------------------------------
# # Example Usage
# # -----------------------------------------------------
# if __name__ == "__main__":
#     model, tokenizer = load_deepseek("4b")

#     prompt = "Explain the difference between supervised and unsupervised learning."
#     response = generate_text(model, tokenizer, prompt)

#     print("\n\n===== MODEL RESPONSE =====\n")
#     print(response)

# from concurrent.futures import ThreadPoolExecutor, TimeoutError
# import logging
# import google.generativeai as genai
# from google.generativeai.types import GenerationConfig

# logger = logging.getLogger(__name__)

# def load_gemini(api_key: str):
#     genai.configure(api_key=api_key)
#     return genai

# def generate_text(client, prompt: str, model: str = "gemini-2.5-pro",
#                   max_tokens: int = 60000, timeout: int =620):
#     def call():
#         # Create the model instance
#         model_instance = genai.GenerativeModel(model)
        
#         # Generate content with configuration
#         resp = model_instance.generate_content(
#             contents = prompt,
#             generation_config = GenerationConfig(
#                 max_output_tokens = max_tokens
#             )
#         )
#         return resp.text

#     try:
#         with ThreadPoolExecutor(max_workers=1) as executor:
#             future = executor.submit(call)
#             return future.result(timeout=timeout)
#     except TimeoutError:
#         logger.error(f"Generation timed out after {timeout} seconds")
#         return None
#     except Exception as e:
#         logger.error(f"Error during Gemini generation: {e}")
#         raise


import logging
from openai import OpenAI  # or whichever OpenAI-compatible SDK you choose
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)

def load_grok(api_key: str):
    return OpenAI(api_key=api_key,
                  base_url="https://api.x.ai/v1")

def generate_text(client, prompt: str, model: str = "grok-4-fast-non-reasoning",
                  timeout: int = 620):
    def call():
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000000,
            response_format={ "type": "json_object" },
        )
        return resp.choices[0].message.content  # adjust depending on client

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call)
            return future.result(timeout=timeout)
    except TimeoutError:
        logger.error(f"Generation timed out after {timeout} seconds")
        return None
    except Exception as e:
        logger.error(f"Error during Grok generation: {e}")
        raise
