# Image-conditioned film colour grading

The project learns one spatially constant grade for an F-Log2 frame. You choose
the reference movie; the system retrieves photographically compatible frames
inside that movie and optimizes a constrained global colour transform.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- An F-Log2/F-Gamut source video. Other camera encodings require a corresponding
  input transform in `colorgrade/color.py`.
- A local corpus of display-referred movie stills, described below.

Install the locked dependencies:

```bash
uv sync
```

## Film corpus

The frame images come from Kaggle's
[Movie Identification Dataset (800 Movies)](https://www.kaggle.com/datasets/asaniczka/movie-identification-dataset-800-movies),
published by `asaniczka`. The image corpus is deliberately excluded from this
Git repository; download it separately and follow the dataset page's licence
and usage conditions.

Place the extracted frames directly under `film_corpus/` using this layout:

```text
film_corpus/10 Things I Hate About You (1999)/frame_0000.jpg
```

The movie directory name is the look name passed to `--look`. It must also
match an `original_title` entry in `movie_titles_acquisition_annotated.csv`
(matching is case-insensitive). By default, only rows whose
`acquisition_type` is `Film` are indexed.

Several different scenes per movie are preferable; the style encoder needs at
least two frames per movie, and training it meaningfully requires at least two
movie directories.

### Corpus provenance

The JPEG corpus originates from the Kaggle dataset linked above. The
`movie_titles_acquisition_annotated.csv` file in this repository is an
additional project-specific annotation used to select photochemically acquired
movies. Its `source_url` column documents camera/film-acquisition research; it
is not the source of the frame images.

## Minimal example

Place an F-Log2/F-Gamut clip at `input.MOV`, prepare at least two movie folders
as above, and run a short smoke-test training:

```bash
# Build a fresh index and train a small style encoder.
uv run python train_style.py \
  --index film_index_demo.pt \
  --output movie_style_encoder_demo.pt \
  --epochs 1 \
  --steps-per-epoch 2 \
  --movies-per-batch 2 \
  --frames-per-movie 2

# Learn the selected look and automatically render a preview and full image.
uv run python train_grade.py input.MOV \
  --look "Moonrise Kingdom (2012)" \
  --index film_index_demo.pt \
  --style-encoder movie_style_encoder_demo.pt \
  --steps 10 \
  --max-side 128
```

The short run only verifies that the pipeline works; it is not expected to
produce a good grade. For a real run, train the reusable style encoder and then
the grade for longer:

```bash
uv run python train_style.py --epochs 20

uv run python train_grade.py input.MOV \
  --look "Moonrise Kingdom (2012)" \
  --steps 800
```

`train_grade.py` writes a checkpoint, a reduced preview, and a full-resolution
render under `outputs/`. It predicts the grade once from a reduced copy of the
complete frame, then applies that same grade to every native-resolution tile.

To reapply an existing checkpoint:

```bash
uv run python apply_grade.py input.MOV \
  --checkpoint outputs/input__Moonrise_Kingdom_2012__t00_00_01__grade.pt
```

## Source tree

```text
colorgrade/
  media.py      image and video loading
  color.py      colour-space mathematics and differentiable pixel operations
  model.py      constrained global grading network
  style.py      content-invariant movie-style encoder
  losses.py     style and distribution objectives
  corpus.py     movie indexing and reference-frame selection

train_style.py  train the reusable movie-style encoder
train_grade.py  learn one grade and render it
apply_grade.py  reapply a saved grade at full resolution
```

Generated model files (`film_index.pt`, `movie_style_encoder.pt`, grade
checkpoints) and source media remain at the repository root or under `outputs/`;
they are data, not source code.
