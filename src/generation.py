"""
Answer generation: retrieved page images + query → Qwen2-VL-2B-Instruct (local, open-source).
2B vision-language model, ~4GB bfloat16, runs on MPS/CPU. No API key required.
"""

from __future__ import annotations

import os
import torch
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

_model = None
_processor = None
_device = None


def load_model(device: str = None):
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor, _device

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    _device = device

    hf_token = os.environ.get("HF_TOKEN")
    print(f"Loading Qwen2-VL-2B on {device}...")

    _processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=hf_token,
    ).to(device).eval()

    print("Qwen2-VL-2B loaded.")
    return _model, _processor, _device


def format_citation(result: Dict[str, Any]) -> str:
    return f"[{result['doc_name']}, p.{result['page_num']}]"


def generate_answer(
    query: str,
    retrieved_pages: List[Dict[str, Any]],
    model,
    processor,
    device: str,
    max_pages: int = 3,
) -> Dict[str, Any]:
    pages = retrieved_pages[:max_pages]
    page_answers = []

    for page in pages:
        img_path = page.get("image_path", "")
        if not img_path or not Path(img_path).exists():
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {
                        "type": "text",
                        "text": (
                            f"This is page {page['page_num']} from NASA report '{page['doc_name']}'.\n"
                            f"Question: {query}\n"
                            "Answer based only on what is visible on this page. Be concise."
                        ),
                    },
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.5,
                no_repeat_ngram_size=4,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = processor.decode(new_tokens, skip_special_tokens=True).strip()

        page_answers.append({
            "page": page,
            "raw_answer": answer,
            "citation": format_citation(page),
        })

    if not page_answers:
        return {
            "answer": "Could not generate an answer — no valid page images found.",
            "citations": [],
            "sources": pages,
            "model": MODEL_ID,
            "tokens_used": 0,
        }

    if len(page_answers) == 1:
        final_answer = page_answers[0]["raw_answer"]
    else:
        parts = [f"From {pa['citation']}:\n{pa['raw_answer']}" for pa in page_answers]
        final_answer = "\n\n".join(parts)

    citations = [pa["citation"] for pa in page_answers]
    final_answer += f"\n\n**Sources:** {', '.join(citations)}"

    return {
        "answer": final_answer,
        "citations": citations,
        "sources": pages,
        "model": MODEL_ID,
        "tokens_used": 0,
    }


def build_generator(api_key: str = None):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, device = load_model(device)

    def answer(query: str, retrieved_pages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        return generate_answer(query, retrieved_pages, model, processor, device, **kwargs)

    return model, answer
