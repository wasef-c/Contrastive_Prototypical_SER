#!/usr/bin/env python3
"""Build the merged MSP-Podcast dataset described in docs/BUILD_MSPP_DATASET.md.

One HuggingFace dataset carrying, per utterance: raw 16 kHz audio, the
transcript, the consensus label and VAD, per-annotator statistics computed from
the raw ratings, and the official Split_Set.

Sources
-------
labels
    /home/rml/Documents/pythontest/mspp/Labels, the November 2024 release.
    `labels_consensus.csv` supplies the consensus columns (its VAD is already
    the plain mean of the per-annotator ratings, so it is taken as released
    rather than recomputed) and the official Split_Set.  `labels_detailed.csv`
    supplies the 649,102 per-annotator rows.
audio and transcript
    cairocode/MSPP_WAV_speaker_split.  Verified to cover exactly the 116,221
    consensus FileNames, one row each, with `audio.path` holding the FileName
    and every file 16 kHz mono PCM_16.  Only its audio and transcript are used:
    its own Split_Set is a speaker-disjoint re-split and is discarded in favour
    of the official one.  The local Audios directory is not usable as a source
    because it holds only 5,942 of the 116,221 files.

Nothing precomputed is carried over.  Feature extraction is the consuming
pipeline's job, and a silent pooling or revision mismatch is expensive to
detect after the fact.

Run as a script; it is not meant to be driven interactively.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Features, Sequence as HFSequence, Value
from huggingface_hub import HfApi, snapshot_download

LABELS_DIR = pathlib.Path("/home/rml/Documents/pythontest/mspp/Labels")
WAV_REPO = "cairocode/MSPP_WAV_speaker_split"
DEFAULT_REPO_ID = "cairocode/MSPP_Full_Annotated"
LABELS_RELEASE = "November 2024"

EXPECTED_ROWS = 116_221
EXPECTED_SPLITS = {"Train": 84_260, "Development": 31_961}

# Official Split_Set value -> HuggingFace split name.  The release contains no
# Test split, so Development becomes validation.
SPLIT_NAMES = {"Train": "train", "Development": "validation"}

# Consensus EmoClass codes.  Kept in full: X (no agreement) is 19% of the
# corpus and is data about ambiguity, so filtering is a training-time decision.
EMOCLASS_TO_LABEL = {
    "N": 0,   # neutral
    "H": 1,   # happy
    "S": 2,   # sad
    "A": 3,   # angry
    "C": 4,   # contempt
    "U": 5,   # surprise
    "D": 6,   # disgust
    "F": 7,   # fear
    "O": 8,   # other
    "X": 9,   # no agreement
}

EMOCLASS_MEANING = {
    "N": "neutral", "H": "happy", "S": "sad", "A": "angry", "C": "contempt",
    "U": "surprise", "D": "disgust", "F": "fear", "O": "other",
    "X": "no agreement",
}

# Canonical primary-emotion vocabulary for label_dist.  The detailed release has
# 1,406 distinct primary strings; everything outside the eight named emotions is
# a rare Other-* variant and collapses into a single Other bucket.
PRIMARY_CLASSES = ["Neutral", "Happy", "Sad", "Angry", "Contempt", "Surprise",
                   "Disgust", "Fear", "Other"]
PRIMARY_OTHER = len(PRIMARY_CLASSES) - 1

# Secondary tags kept as subtypes, in the order used by subtype_dist.  These are
# the sixteen most frequent tags in the release; each appears at least 19k
# times, and the next most frequent is Other-Curious at 1,082.
SUBTYPES = ["Neutral", "Happy", "Concerned", "Frustrated", "Angry", "Sad",
            "Amused", "Annoyed", "Excited", "Disappointed", "Contempt",
            "Surprise", "Disgust", "Confused", "Depressed", "Fear"]

# Misspellings in the release that name an existing tag.
SUBTYPE_ALIASES = {"Dissapointed": "Disappointed"}

TARGET_SHARD_BYTES = 450 * 1024 * 1024


def canonical_primary(primary: str) -> int:
    """Map a raw primary-emotion string onto the canonical vocabulary.

    Args:
        primary: primary label as written in labels_detailed.csv.

    Returns:
        Index into PRIMARY_CLASSES; PRIMARY_OTHER for anything unnamed.
    """
    try:
        return PRIMARY_CLASSES.index(primary)
    except ValueError:
        return PRIMARY_OTHER


def parse_detailed_labels(labels_dir: pathlib.Path) -> Dict[str, dict]:
    """Accumulate per-annotator statistics from labels_detailed.csv.

    Fields per row are `worker; primary; secondary_tags; A:; V:; D:`.  The
    secondary field is absent on some rows and is detected by the absence of a
    colon, since A/V/D always carry one.  Rows are quoted, so csv.reader is
    required rather than a split on commas.

    Args:
        labels_dir: directory holding labels_detailed.csv.

    Returns:
        FileName -> dict with `counts` (per canonical primary), `n`, `avd`
        (list of raw V, A, D triples) and `sub` (per subtype counts).
    """
    sub_index = {s: i for i, s in enumerate(SUBTYPES)}
    per_file: Dict[str, dict] = {}

    with open(labels_dir / "labels_detailed.csv", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            file_name, detail = row[0], row[1]
            parts = [p.strip() for p in detail.split(";") if p.strip()]
            if len(parts) < 2:
                continue

            primary = parts[1]
            secondary = parts[2] if len(parts) > 2 and ":" not in parts[2] else ""

            avd: Dict[str, float] = {}
            for part in parts:
                if len(part) > 2 and part[0] in "AVD" and part[1] == ":":
                    try:
                        avd[part[0]] = float(part[2:])
                    except ValueError:
                        pass

            entry = per_file.get(file_name)
            if entry is None:
                entry = {
                    "counts": np.zeros(len(PRIMARY_CLASSES), dtype=np.float64),
                    "n": 0,
                    "avd": [],
                    "sub": np.zeros(len(SUBTYPES), dtype=np.float64),
                }
                per_file[file_name] = entry

            entry["counts"][canonical_primary(primary)] += 1.0
            entry["n"] += 1
            if len(avd) == 3:
                entry["avd"].append((avd["V"], avd["A"], avd["D"]))
            for tag in secondary.split(","):
                tag = tag.strip()
                tag = SUBTYPE_ALIASES.get(tag, tag)
                index = sub_index.get(tag)
                if index is not None:
                    entry["sub"][index] += 1.0

    return per_file


def summarise_annotations(entry: dict) -> dict:
    """Reduce one utterance's accumulated ratings to the stored statistics.

    Args:
        entry: value from parse_detailed_labels.

    Returns:
        Dict with n_annotators, label_dist, annot_entropy, annot_vad_mean and
        annot_vad_std.  VAD dispersion is a population std: with five raters the
        sample correction is mostly noise.
    """
    total = max(entry["n"], 1)
    dist = entry["counts"] / total
    positive = dist[dist > 0]
    entropy = float(-(positive * np.log(positive)).sum())

    if entry["avd"]:
        ratings = np.asarray(entry["avd"], dtype=np.float64)
        vad_mean = ratings.mean(axis=0)
        vad_std = ratings.std(axis=0)
    else:
        vad_mean = np.zeros(3, dtype=np.float64)
        vad_std = np.zeros(3, dtype=np.float64)

    return {
        "n_annotators": int(entry["n"]),
        "label_dist": dist.astype(np.float32).tolist(),
        "annot_entropy": entropy,
        "annot_vad_mean": vad_mean.astype(np.float32).tolist(),
        "annot_vad_std": vad_std.astype(np.float32).tolist(),
        "subtype_dist": (entry["sub"] / total).astype(np.float32).tolist(),
    }


def load_consensus(labels_dir: pathlib.Path) -> Dict[str, dict]:
    """Read labels_consensus.csv keyed on FileName.

    The consensus VAD columns are the plain mean of the per-annotator ratings
    and are taken as released rather than recomputed.

    Args:
        labels_dir: directory holding labels_consensus.csv.

    Returns:
        FileName -> dict of consensus fields.
    """
    consensus: Dict[str, dict] = {}
    with open(labels_dir / "labels_consensus.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            consensus[row["FileName"]] = {
                "EmoClass": row["EmoClass"],
                "label": EMOCLASS_TO_LABEL[row["EmoClass"]],
                "EmoAct": float(row["EmoAct"]),
                "EmoVal": float(row["EmoVal"]),
                "EmoDom": float(row["EmoDom"]),
                "SpkrID": int(row["SpkrID"]),
                "Gender": row["Gender"],
                "Split_Set": row["Split_Set"],
            }
    return consensus


def build_features() -> Features:
    """Describe the output schema.

    Returns:
        HuggingFace Features for the merged dataset.
    """
    return Features({
        "FileName": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "transcript": Value("string"),
        "EmoClass": Value("string"),
        "label": Value("int64"),
        "EmoAct": Value("float32"),
        "EmoVal": Value("float32"),
        "EmoDom": Value("float32"),
        "SpkrID": Value("int64"),
        "Gender": Value("string"),
        "Split_Set": Value("string"),
        "n_annotators": Value("int32"),
        "label_dist": HFSequence(Value("float32"), length=len(PRIMARY_CLASSES)),
        "annot_entropy": Value("float32"),
        "annot_vad_mean": HFSequence(Value("float32"), length=3),
        "annot_vad_std": HFSequence(Value("float32"), length=3),
        "subtype_dist": HFSequence(Value("float32"), length=len(SUBTYPES)),
    })


def wav_shard_paths(source: Optional[pathlib.Path]) -> List[pathlib.Path]:
    """Locate the parquet shards carrying audio and transcript.

    Args:
        source: local directory of parquet files, or None to resolve WAV_REPO
            from the HuggingFace cache (downloading it if absent).

    Returns:
        Sorted list of parquet paths.
    """
    if source is not None:
        root = source
    else:
        root = pathlib.Path(snapshot_download(WAV_REPO, repo_type="dataset"))
    shards = sorted(root.rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {root}")
    return shards


class ShardWriter:
    """Write one split to size-bounded parquet shards.

    Rows are buffered until the accumulated audio exceeds a target size, so peak
    memory stays near one shard rather than one split.
    """

    def __init__(self, out_dir: pathlib.Path, split: str, schema: pa.Schema,
                 target_bytes: int) -> None:
        """Create a writer for one split.

        Args:
            out_dir: directory to write `<split>-NNNNN.parquet` into.
            split: HuggingFace split name.
            schema: arrow schema, carrying the HuggingFace feature metadata.
            target_bytes: approximate uncompressed audio per shard.
        """
        self.out_dir = out_dir
        self.split = split
        self.schema = schema
        self.target_bytes = target_bytes
        self.buffer: List[dict] = []
        self.buffered_bytes = 0
        self.shard_index = 0
        self.rows = 0

    def add(self, row: dict, nbytes: int) -> None:
        """Buffer one row, flushing when the target shard size is reached.

        Args:
            row: row matching the output schema.
            nbytes: size of the row's audio, used for shard sizing.
        """
        self.buffer.append(row)
        self.buffered_bytes += nbytes
        self.rows += 1
        if self.buffered_bytes >= self.target_bytes:
            self.flush()

    def flush(self) -> None:
        """Write the buffered rows as one parquet shard."""
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        path = self.out_dir / f"{self.split}-{self.shard_index:05d}.parquet"
        pq.write_table(table, path, compression="snappy")
        print(f"  wrote {path.name}  rows={table.num_rows} "
              f"size={path.stat().st_size / 1e6:.0f} MB", flush=True)
        self.buffer = []
        self.buffered_bytes = 0
        self.shard_index += 1


def audio_is_16k_mono(raw: bytes) -> Tuple[bool, int, int]:
    """Check that an encoded audio blob is 16 kHz mono.

    Args:
        raw: encoded audio bytes.

    Returns:
        Tuple of (ok, sample_rate, channels); sample_rate is -1 if unreadable.
    """
    try:
        info = sf.info(io.BytesIO(raw))
    except Exception:
        return False, -1, -1
    return (info.samplerate == 16000 and info.channels == 1,
            info.samplerate, info.channels)


def build(labels_dir: pathlib.Path, wav_source: Optional[pathlib.Path],
          out_dir: pathlib.Path, target_bytes: int) -> Dict[str, int]:
    """Merge the sources and write the dataset to parquet shards.

    Args:
        labels_dir: directory holding the official label CSVs.
        wav_source: local parquet directory for audio, or None for the hub.
        out_dir: destination directory; `data/` is created inside it.
        target_bytes: approximate audio bytes per output shard.

    Returns:
        Row count per HuggingFace split.

    Raises:
        SystemExit: if any consistency check fails.
    """
    print("parsing labels_detailed.csv ...", flush=True)
    detailed = parse_detailed_labels(labels_dir)
    print(f"  {len(detailed)} utterances with per-annotator ratings", flush=True)

    print("reading labels_consensus.csv ...", flush=True)
    consensus = load_consensus(labels_dir)
    print(f"  {len(consensus)} consensus rows", flush=True)

    missing_detail = set(consensus) - set(detailed)
    if missing_detail:
        raise SystemExit(f"{len(missing_detail)} consensus rows have no "
                         f"per-annotator ratings, e.g. {sorted(missing_detail)[:3]}")

    stats = {name: summarise_annotations(entry) for name, entry in detailed.items()}
    del detailed

    features = build_features()
    schema = features.arrow_schema
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    writers = {split: ShardWriter(data_dir, split, schema, target_bytes)
               for split in sorted(set(SPLIT_NAMES.values()))}

    seen: set = set()
    missing_transcript = 0
    missing_audio = 0
    bad_format: List[Tuple[str, int, int]] = []

    shards = wav_shard_paths(wav_source)
    print(f"streaming audio and transcript from {len(shards)} source shards ...",
          flush=True)

    for shard in shards:
        handle = pq.ParquetFile(shard)
        for group in range(handle.num_row_groups):
            rows = handle.read_row_group(
                group, columns=["audio", "transcript"]).to_pylist()
            for row in rows:
                audio = row["audio"]
                file_name = audio["path"]
                meta = consensus.get(file_name)
                if meta is None:
                    raise SystemExit(f"{file_name} is not in labels_consensus.csv")
                if file_name in seen:
                    raise SystemExit(f"{file_name} appears twice in the audio source")
                seen.add(file_name)

                raw = audio["bytes"] or b""
                if not raw:
                    missing_audio += 1
                else:
                    ok, rate, channels = audio_is_16k_mono(raw)
                    if not ok:
                        bad_format.append((file_name, rate, channels))

                transcript = row["transcript"] or ""
                if not transcript.strip():
                    missing_transcript += 1

                out_row = {
                    "FileName": file_name,
                    "audio": {"bytes": raw, "path": file_name},
                    "transcript": transcript,
                    **{k: meta[k] for k in ("EmoClass", "label", "EmoAct",
                                            "EmoVal", "EmoDom", "SpkrID",
                                            "Gender", "Split_Set")},
                    **stats[file_name],
                }
                writers[SPLIT_NAMES[meta["Split_Set"]]].add(out_row, len(raw))
        print(f"  {shard.name}: {len(seen)} rows so far", flush=True)

    for writer in writers.values():
        writer.flush()

    counts = {split: writer.rows for split, writer in writers.items()}
    print(f"\nrows per split: {counts}")
    print(f"missing audio: {missing_audio}")
    print(f"missing transcript: {missing_transcript}")
    print(f"audio not 16 kHz mono: {len(bad_format)}")

    problems: List[str] = []
    if len(seen) != EXPECTED_ROWS:
        problems.append(f"wrote {len(seen)} rows, expected {EXPECTED_ROWS}")
    if len(seen) != sum(counts.values()):
        problems.append("row total does not match the per-split totals")
    for split_set, expected in EXPECTED_SPLITS.items():
        got = counts[SPLIT_NAMES[split_set]]
        if got != expected:
            problems.append(f"{split_set}: wrote {got} rows, expected {expected}")
    if missing_audio:
        problems.append(f"{missing_audio} rows have no audio")
    if bad_format:
        problems.append(f"{len(bad_format)} files are not 16 kHz mono, "
                        f"e.g. {bad_format[:3]}")
    if problems:
        raise SystemExit("verification failed:\n  " + "\n  ".join(problems))

    print("verification passed", flush=True)
    write_card(out_dir, counts, missing_transcript)
    return counts


def write_card(out_dir: pathlib.Path, counts: Dict[str, int],
               missing_transcript: int) -> None:
    """Write the dataset card, including provenance and field orderings.

    Args:
        out_dir: dataset directory to write README.md into.
        counts: row count per HuggingFace split.
        missing_transcript: number of rows with an empty transcript.
    """
    class_table = "\n".join(
        f"| {code} | {EMOCLASS_MEANING[code]} | {label} |"
        for code, label in EMOCLASS_TO_LABEL.items())

    card = f"""---
license: other
language:
- en
task_categories:
- audio-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
---

# MSP-Podcast, merged with per-annotator statistics

Every utterance in the MSP-Podcast release carries its raw 16 kHz audio, its
transcript, the consensus label and VAD, and statistics computed from the raw
per-annotator ratings.

## Provenance

- Audio and transcript: `{WAV_REPO}`, verified to cover exactly the
  {EXPECTED_ROWS} consensus FileNames, one row each, all 16 kHz mono PCM_16.
- Labels: the official MSP-Podcast release of {LABELS_RELEASE}
  (`labels_consensus.csv`, `labels_detailed.csv`), 649,102 per-annotator rows
  over {EXPECTED_ROWS} utterances, mean 5.59 annotators each (min 5, max 32).
- Splits are the official `Split_Set`: Train ({EXPECTED_SPLITS['Train']}) and
  Development ({EXPECTED_SPLITS['Development']}), exposed as `train` and
  `validation`. The release contains no Test split. The speaker-disjoint
  re-split carried by the audio source was discarded.
- No precomputed encoder features are carried, so the consuming pipeline
  cannot silently inherit a different pooling or model revision.
- Rows written: {json.dumps(counts)}. Rows with an empty transcript:
  {missing_transcript}.

## Fields

| field | type | meaning |
|---|---|---|
| `FileName` | str | join key, e.g. `MSP-PODCAST_0002_0033.wav` |
| `audio` | Audio | 16 kHz mono |
| `transcript` | str | |
| `EmoClass` | str | consensus class code |
| `label` | int | integer form of `EmoClass` |
| `EmoAct` / `EmoVal` / `EmoDom` | float | consensus VAD, 1-7 scale, as released |
| `SpkrID` | int | |
| `Gender` | str | Male / Female |
| `Split_Set` | str | official Train / Development |
| `n_annotators` | int | |
| `label_dist` | {len(PRIMARY_CLASSES)} floats | fraction of annotators per canonical primary emotion |
| `annot_entropy` | float | Shannon entropy of `label_dist`, natural log |
| `annot_vad_mean` | 3 floats | mean of the raw per-annotator ratings, (V, A, D) |
| `annot_vad_std` | 3 floats | population std of the same |
| `subtype_dist` | {len(SUBTYPES)} floats | fraction of annotators applying each secondary tag |

Consensus VAD in the release is already the plain mean of the per-annotator
ratings, so it is taken as released; only the dispersion is computed here.

## Class vocabulary

All ten consensus classes are kept, including `X` (no agreement), which is 19%
of the corpus. An utterance annotators could not agree on is data about
ambiguity, so filtering it is a training-time decision.

| code | meaning | `label` |
|---|---|---|
{class_table}

## `label_dist` ordering

{json.dumps(PRIMARY_CLASSES)}

`labels_detailed.csv` has 1,406 distinct primary strings. The eight named
emotions are kept as written; every other string, nearly all of them rare
`Other-*` variants, is counted in the single `Other` bucket.

## `subtype_dist` ordering

{json.dumps(SUBTYPES)}

These are annotator-supplied subtypes inside the coarse classes: Amused and
Excited within happy, Depressed and Disappointed within sad, Frustrated and
Annoyed within angry, Concerned alongside neutral. The release misspelling
`Dissapointed` (1,598 occurrences) is merged into `Disappointed`. Tags outside
this list are rare (the next most frequent is `Other-Curious` at 1,082) and are
not counted.

## Licence

MSP-Podcast is distributed under its own restricted licence by UT Dallas. This
repository is a derived packaging for internal research use and inherits those
terms.
"""
    (out_dir / "README.md").write_text(card)
    print(f"wrote {out_dir / 'README.md'}", flush=True)


def push(out_dir: pathlib.Path, repo_id: str, private: bool) -> None:
    """Create the dataset repo if needed and upload the built directory.

    Args:
        out_dir: directory holding README.md and data/.
        repo_id: destination repo, e.g. cairocode/MSPP_Full_Annotated.
        private: whether to create the repo private.
    """
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Merged MSP-Podcast: audio, transcript, consensus and "
                       "per-annotator statistics",
    )
    print(f"pushed to https://huggingface.co/datasets/{repo_id}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, build the dataset and optionally push it.

    Args:
        argv: command line arguments, defaulting to sys.argv[1:].

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=pathlib.Path, default=LABELS_DIR,
                        help="directory holding the official label CSVs")
    parser.add_argument("--wav-source", type=pathlib.Path, default=None,
                        help="local directory of audio parquet shards; "
                             f"defaults to the {WAV_REPO} snapshot")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True,
                        help="directory to build the dataset in")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help="destination HuggingFace dataset repo")
    parser.add_argument("--shard-mb", type=int,
                        default=TARGET_SHARD_BYTES // (1024 * 1024),
                        help="approximate audio megabytes per output shard")
    parser.add_argument("--public", action="store_true",
                        help="create the repo public rather than private")
    parser.add_argument("--push", action="store_true",
                        help="upload after the verification passes")
    args = parser.parse_args(argv)

    build(args.labels_dir, args.wav_source, args.out_dir,
          args.shard_mb * 1024 * 1024)
    if args.push:
        push(args.out_dir, args.repo_id, private=not args.public)
    else:
        print("built but not pushed; re-run with --push to upload", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
