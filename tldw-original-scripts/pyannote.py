from pyannote.audio import Pipeline
import torch
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization").to(torch.device("cuda"))

# 4. apply pretrained pipeline
diarization = pipeline("lex.wav", num_speakers=2)

# 5. print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
