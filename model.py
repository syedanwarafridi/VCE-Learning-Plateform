from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

MODEL_NAME = "ibm-granite/granite-3.3-8b-base"

print("🔄 Starting server and loading Granite model...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto"
)

model.eval()

print("✅ Granite model loaded and ready.")


# ============================================================
# Inference Function
# ============================================================
def granite_generate(
    system_prompt: str,
    user_prompt: str,
    context: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9
):
    prompt = f"{system_prompt}\n\n"
    if context:
        prompt += f"CONTEXT:\n{context}\n\n"
    prompt += f"USER:\n{user_prompt}\n\nASSISTANT:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()


# ============================================================
# API Endpoint
# ============================================================
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json

        system_prompt = data.get("system_prompt", "You are a helpful assistant.")
        user_prompt = data.get("user_prompt", "")
        context = data.get("context", "")

        if not user_prompt:
            return jsonify({"error": "user_prompt is required"}), 400

        output = granite_generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context
        )

        return jsonify({"output": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# App Entry Point
# ============================================================
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=PORT)
