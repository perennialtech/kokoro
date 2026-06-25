from pathlib import Path

from kokoro import KModel, export_onnx


def export_kokoro_to_onnx():
    print("Loading PyTorch model...")
    model = KModel().eval()

    output_dir = "onnx"
    print(f"Exporting ONNX models to '{output_dir}/'...")

    export_onnx(model, output_dir)

    print("\nExport successful! Created the following files:")
    for file in Path(output_dir).iterdir():
        print(f"  - {file}")


if __name__ == "__main__":
    export_kokoro_to_onnx()
