from kokoro import KModel


def export_kokoro_to_onnx():
    print("Loading PyTorch model...")
    model = KModel()

    # 2. Export to ONNX
    output_dir = "onnx"
    print(f"Exporting ONNX models to '{output_dir}/'...")

    paths = model.export_onnx(output_dir=output_dir)

    print("\nExport successful! Created the following files:")
    for prefix, files in paths.items():
        print(f"\n{prefix}:")
        print(f"  - {files}")


if __name__ == "__main__":
    export_kokoro_to_onnx()
