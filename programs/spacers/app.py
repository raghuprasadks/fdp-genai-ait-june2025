import spaces
import gradio as gr
from sacremoses import MosesPunctNormalizer
from stopes.pipelines.monolingual.utils.sentence_split import get_split_algo
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flores import code_mapping
import platform
import torch
import nltk
from functools import lru_cache

nltk.download("punkt_tab")

REMOVED_TARGET_LANGUAGES = {"Ligurian", "Lombard", "Sicilian"}

# ✅ Dynamic CUDA check - use GPU only if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MODEL_NAME = "facebook/nllb-200-3.3B"

code_mapping = dict(sorted(code_mapping.items(), key=lambda item: item[0]))
flores_codes = list(code_mapping.keys())
target_languages = [language for language in flores_codes if not language in REMOVED_TARGET_LANGUAGES]

def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print(f"Model loaded in {device}")
    return model

model = load_model()

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
punct_normalizer = MosesPunctNormalizer(lang="en")

@lru_cache(maxsize=202)
def get_language_specific_sentence_splitter(language_code):
    short_code = language_code[:3]
    splitter = get_split_algo(short_code, "default")
    return splitter

@lru_cache(maxsize=100)
def translate(text: str, src_lang: str, tgt_lang: str):
    if not src_lang:
        raise gr.Error("The source language is empty! Please choose it in the dropdown list.")
    if not tgt_lang:
        raise gr.Error("The target language is empty! Please choose it in the dropdown list.")
    return _translate(text, src_lang, tgt_lang)

@spaces.GPU
def _translate(text: str, src_lang: str, tgt_lang: str):
    src_code = code_mapping[src_lang]
    tgt_code = code_mapping[tgt_lang]
    tokenizer.src_lang = src_code
    tokenizer.tgt_lang = tgt_code

    text = punct_normalizer.normalize(text)
    paragraphs = text.split("\n")
    translated_paragraphs = []

    for paragraph in paragraphs:
        splitter = get_language_specific_sentence_splitter(src_code)
        sentences = list(splitter(paragraph))
        translated_sentences = []

        for sentence in sentences:
            input_tokens = (
                tokenizer(sentence, return_tensors="pt")
                .input_ids[0]
                .cpu()
                .numpy()
                .tolist()
            )
            translated_chunk = model.generate(
                input_ids=torch.tensor([input_tokens]).to(device),
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=len(input_tokens) + 50,
                num_return_sequences=1,
                num_beams=5,
                no_repeat_ngram_size=4,
                renormalize_logits=True,
            )
            translated_chunk = tokenizer.decode(
                translated_chunk[0], skip_special_tokens=True
            )
            translated_sentences.append(translated_chunk)

        translated_paragraph = " ".join(translated_sentences)
        translated_paragraphs.append(translated_paragraph)

    return "\n".join(translated_paragraphs)

description = """<div style="text-align: center;">
    <img src="https://huggingface.co/spaces/UNESCO/nllb/resolve/main/UNESCO_META_HF_BANNER.png" alt="UNESCO Meta Hugging Face Banner" style="max-width: 800px; width: 100%; margin: 0 auto;">
    <h1 style="color: #0077be;">UNESCO Language Translator, powered by Meta and Hugging Face</h1></div>
    UNESCO, Meta, and Hugging Face have come together to create an accessible, high-quality translation experience in 200 languages."""
disclaimer = """## Disclaimer
(This section remains unchanged)
"""

examples_inputs = [["The United Nations Educational, Scientific and Cultural Organization is a specialized agency of the United Nations with the aim of promoting world peace and security through international cooperation in education, arts, sciences and culture.", "English", "Ayacucho Quechua"],]

with gr.Blocks() as demo:
    gr.Markdown(description)
    with gr.Row():
        src_lang = gr.Dropdown(label="Source Language", choices=flores_codes)
        target_lang = gr.Dropdown(label="Target Language", choices=target_languages)
    with gr.Row():
        input_text = gr.Textbox(label="Input Text", lines=6)
    with gr.Row():
        btn = gr.Button("Translate text")
    with gr.Row():
        output = gr.Textbox(label="Output Text", lines=6)
    btn.click(
        translate,
        inputs=[input_text, src_lang, target_lang],
        outputs=output,
    )
    examples = gr.Examples(examples=examples_inputs, inputs=[input_text, src_lang, target_lang], fn=translate, outputs=output, cache_examples=True)
    with gr.Row():
        gr.Markdown(disclaimer)
demo.launch()
