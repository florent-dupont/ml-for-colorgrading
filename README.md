# Image-conditioned film colour grading

This project learns a spatially constant colour grade for an F-Log2/F-Gamut
frame. A movie is selected as the reference look; the model retrieves compatible
frames from that movie and learns a constrained global colour transform.

## Installation

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Film corpus

Download Kaggle's
[Movie Identification Dataset (800 Movies)](https://www.kaggle.com/datasets/asaniczka/movie-identification-dataset-800-movies)
and place the extracted frames under `film_corpus/`:

```text
film_corpus/Batman (1989)/frame_0000.jpg
```


## Usage

Train the reusable movie-style encoder once:

```bash
python train_style.py --epochs 20
```

Train a grade for an F-Log2/F-Gamut source and render the result:

```bash
python train_grade.py input.MOV \
  --look "Moonrise Kingdom (2012)" \
  --steps 800
```

This creates a model checkpoint, preview, and full-resolution render under
`outputs/`. The grade is predicted once from the complete frame and applied
unchanged at full resolution.

Reapply an existing checkpoint:

```bash
python apply_grade.py input.MOV \
  --checkpoint outputs/input__Moonrise_Kingdom_2012__t00_00_01__grade.pt
```

## Project structure

```text
colorgrade/       model, colour science, losses, corpus and media utilities
train_style.py    train the reusable movie-style encoder
train_grade.py    learn and render a grade
apply_grade.py    apply a saved grade at full resolution
```
