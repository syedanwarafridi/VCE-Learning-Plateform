import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------------------------------
# Configure Logging
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Load Tokenizer
# -----------------------------------------------------
def load_tokenizer(model_name: str):
    try:
        logger.info(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info("Tokenizer loaded successfully.")
        return tokenizer

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
def load_model(model_name: str,
               device: str = "cuda",
               dtype: torch.dtype = torch.float16):
    try:
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {device}, Precision: {dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )

        logger.info("Model loaded successfully.")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# -----------------------------------------------------
# Text Generation
# -----------------------------------------------------
def generate_text(model, tokenizer, prompt: str, max_tokens: int = 200):
    try:
        logger.info("Preparing input for generation...")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logger.info("Input tokenization successful.")

        logger.info("Generating response...")
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )

        logger.info("Generation completed.")
        return tokenizer.decode(output[0], skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise


# -----------------------------------------------------
# Wrapper for 7B or 8B
# -----------------------------------------------------
def load_deepseek(model_type: str = "7b",
                  device: str = "cuda",
                  dtype: torch.dtype = torch.float16):
    try:
        model_map = {
            "7b": "deepseek-ai/deepseek-llm-7b",
            "8b": "deepseek-ai/deepseek-llm-8b"
        }

        if model_type.lower() not in model_map:
            raise ValueError("model_type must be '7b' or '8b'")

        model_name = model_map[model_type.lower()]

        logger.info(f"Selected model: {model_type.upper()} ({model_name})")

        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name, device, dtype)

        logger.info("DeepSeek model and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to initialize DeepSeek model: {e}")
        raise
