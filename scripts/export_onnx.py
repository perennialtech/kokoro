from kokoro import KModel

model = KModel(repo_id="hexgrad/Kokoro-82M")

print("Exporting text and duration model...")
model.export_text_duration_onnx(
    path="onnx/kokoro_text_duration.onnx",
    batch_size=1,
    text_bucket=512,  # Max expected token length
    opset=18,
)

print("Exporting acoustic vocoder model...")
model.export_acoustic_vocoder_onnx(
    path="onnx/kokoro_acoustic_vocoder.onnx",
    batch_size=1,
    frame_bucket=512,  # Max expected audio frame length
    opset=18,
)

print("ONNX export complete!")
