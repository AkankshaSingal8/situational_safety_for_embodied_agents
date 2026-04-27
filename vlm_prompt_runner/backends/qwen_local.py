import logging
import os
import time

from vlm_prompt_runner.backends.base import VLMBackend

logger = logging.getLogger(__name__)

QWEN_MODELS = {
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2-vl-7b":   "Qwen/Qwen2-VL-7B-Instruct",
    "qwen3-vl-8b":   "Qwen/Qwen3-VL-8B",
}


class QwenLocalBackend(VLMBackend):
    """Loads a Qwen VLM checkpoint locally via transformers and runs inference."""

    def __init__(self, model_key: str = "qwen2.5-vl-7b",
                 device: str = "auto", load_in_4bit: bool = False):
        import torch
        from transformers import AutoProcessor

        if model_key not in QWEN_MODELS:
            raise ValueError(
                f"Unknown Qwen model key: {model_key!r}. "
                f"Choose from {list(QWEN_MODELS)}"
            )

        hf_id = QWEN_MODELS[model_key]
        logger.info(f"Loading {hf_id} ...")

        if model_key.startswith("qwen2.5"):
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        elif model_key.startswith("qwen2"):
            from transformers import Qwen2VLForConditionalGeneration as ModelCls
        else:
            from transformers import AutoModelForImageTextToText as ModelCls

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        load_kwargs: dict = dict(torch_dtype=dtype, device_map=device)

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            )

        t0 = time.time()
        self.model = ModelCls.from_pretrained(hf_id, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        logger.info(f"Loaded in {time.time()-t0:.1f}s  (class={ModelCls.__name__})")

    def generate(self, prompt: str, image_paths: list[str],
                 max_new_tokens: int = 1024) -> str:
        import torch
        from PIL import Image as PILImage

        content = []
        pil_images = []
        for p in image_paths:
            if p and os.path.exists(p):
                img = PILImage.open(p).convert("RGB")
                pil_images.append(img)
                content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        if pil_images:
            inputs = self.processor(
                text=[prompt_text], images=pil_images,
                return_tensors="pt", padding=True,
            )
        else:
            inputs = self.processor(
                text=[prompt_text], return_tensors="pt", padding=True,
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
