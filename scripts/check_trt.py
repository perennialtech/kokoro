import torch

from kokoro import KModel, KokoroTRTBackend, KPipeline
from kokoro.runtime import expand_frames, normalize_requests

torch.set_printoptions(precision=6, sci_mode=False)

repo_id = "hexgrad/Kokoro-82M"
artifact_dir = "./build"

text = "Hello from TensorRT Kokoro."
voice = "af_heart"

model = KModel(repo_id=repo_id).eval().cuda()
pt_backend = model.inference_backend()

pipeline = KPipeline(
    lang_code="a",
    repo_id=model.repo_id,
    vocab=model.vocab,
    context_length=model.context_length,
)
pipeline.set_voice_target("cuda", torch.float32)

prepared = next(pipeline.prepare(text, voice=voice, speed=1.0))
request = normalize_requests(
    prepared=prepared,
    device=torch.device("cuda"),
    ref_dtype=torch.float32,
)[0]

with torch.inference_mode():
    duration_float, duration_hidden, text_hidden = pt_backend.text_duration(
        request.input_ids,
        request.ref_s,
        request.speed,
    )

    frame_item = expand_frames(duration_float, duration_hidden, text_hidden)

    asr = frame_item.asr.unsqueeze(0)
    en = frame_item.en.unsqueeze(0)
    ref_s = request.ref_s

    f0, n = pt_backend.acoustic_vocoder.predict_f0n(en, ref_s)
    har = model.compute_harmonic_features(f0)

    print("synthesis frames:", frame_item.synthesis_frame_length)
    print("asr:", tuple(asr.shape))
    print("en:", tuple(en.shape))
    print("f0:", tuple(f0.shape))
    print("n:", tuple(n.shape))
    print("har:", tuple(har.shape))
    print("ref_s:", tuple(ref_s.shape))

    pt_audio = pt_backend.acoustic_vocoder.forward_with_f0n(asr, f0, n, ref_s, har)

    trt_backend = KokoroTRTBackend(
        model,
        artifact_dir=artifact_dir,
        fallback_to_pytorch=False,
    )

    trt_audio = trt_backend._decode_with_trt(asr, f0, n, ref_s, har)

    pt_audio = pt_audio.float().reshape(-1)
    trt_audio = trt_audio.float().reshape(-1)

    min_len = min(pt_audio.numel(), trt_audio.numel())
    pt_audio = pt_audio[:min_len]
    trt_audio = trt_audio[:min_len]

    err = pt_audio - trt_audio

    rms_pt = pt_audio.square().mean().sqrt()
    rms_err = err.square().mean().sqrt()
    max_abs = err.abs().max()

    snr = 20 * torch.log10(rms_pt / torch.clamp(rms_err, min=1e-12))

    print("pt audio:", tuple(pt_audio.shape))
    print("trt audio:", tuple(trt_audio.shape))
    print("max abs err:", max_abs.item())
    print("rms pt:", rms_pt.item())
    print("rms err:", rms_err.item())
    print("SNR dB:", snr.item())
