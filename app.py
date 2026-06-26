import os
import time

import gradio as gr
import numpy as np

from kokoro import KokoroTRT
from kokoro.pipeline import LANGUAGE_CODES

ARTIFACT_DIR = os.environ.get("KOKORO_TRT_ARTIFACT_DIR", "./build")
tts = KokoroTRT(ARTIFACT_DIR)


def generate_audio(text, language, voice, speed):
    start = time.perf_counter()

    chunks = [
        result.audio.detach().cpu().numpy()
        for result in tts.synthesize(text, voice, language, speed)
    ]

    if not chunks:
        return None, "No audio generated."

    audio = np.concatenate(chunks)
    duration = time.perf_counter() - start
    audio_dur = len(audio) / 24000
    rtf = duration / audio_dur if audio_dur > 0 else 0

    timing_info = (
        f"Generated {audio_dur:.2f}s of audio in {duration:.3f}s (RTF: {rtf:.3f})"
    )

    return (24000, audio), timing_info


with gr.Blocks() as app:
    gr.Markdown("# Kokoro TensorRT TTS")

    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(
                label="Text",
                lines=5,
                value="Hello from Kokoro running through TensorRT.",
            )
            lang_in = gr.Dropdown(
                label="Language",
                choices=[(name, code) for code, name in LANGUAGE_CODES.items()],
                value="a",
            )
            voice_in = gr.Textbox(label="Voice", value="af_heart")
            speed_in = gr.Slider(
                label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1
            )
            submit_btn = gr.Button("Generate")

        with gr.Column():
            audio_out = gr.Audio(label="Synthesized Audio", type="numpy")
            timing_out = gr.Textbox(label="Timings", interactive=False)

    submit_btn.click(
        generate_audio,
        inputs=[text_in, lang_in, voice_in, speed_in],
        outputs=[audio_out, timing_out],
    )

if __name__ == "__main__":
    app.launch()
