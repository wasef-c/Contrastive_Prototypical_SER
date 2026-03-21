#!/usr/bin/env python3
"""
Diagnostic: check CMUMOSEI and SAMSEMO dataset columns
for preextracted vs audio-only HF paths.
"""
from datasets import load_dataset

datasets_to_check = [
    ("CMUMOSEI preextracted", "cairocode/CMUMOSEI_Emotion2Vec_PrecomputedEncodings"),
    ("CMUMOSEI audio",        "cairocode/cmu_mosei_wav_2"),
    ("SAMSEMO preextracted",  "cairocode/SAMSEMO_Emotion2Vec_PrecomputedEncodings"),
    ("SAMSEMO audio",         "cairocode/samsemo-audio"),
]

for name, path in datasets_to_check:
    print(f"\n{'='*60}")
    print(f"{name}: {path}")
    try:
        ds = load_dataset(path, split="train", trust_remote_code=True)
        print(f"  Columns: {ds.column_names}")
        print(f"  Size:    {len(ds)}")
        item = ds[0]
        print(f"  First item:")
        for k, v in item.items():
            if k == "audio":
                print(f"    {k}: sr={v.get('sampling_rate','?')}, len={len(v.get('array',[]))}")
            elif k == "emotion2vec_features":
                print(f"    {k}: list of {len(v)} items")
            elif isinstance(v, (str, int, float)):
                print(f"    {k}: {repr(v)[:100]}")
            else:
                print(f"    {k}: {type(v).__name__}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone.")
