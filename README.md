# Image-conditioned film colour grading

The project learns one spatially constant grade for an F-Log2 frame. You choose
the reference movie; the system retrieves photographically compatible frames
inside that movie and optimizes a constrained global colour transform.

## Commands

```bash
# Run once, or whenever the film corpus changes substantially.
uv run python train_style.py --epochs 20

# Train a grade and automatically render preview + full resolution.
uv run python train_grade.py input1.MOV \
  --look "Moonrise Kingdom (2012)" --steps 800

# Reapply a saved grade.
uv run python apply_grade.py input1.MOV \
  --checkpoint outputs/input1__Moonrise_Kingdom_2012__t00_00_01__grade.pt
```

Outputs are written under `outputs/` as `__grade.pt`, `__preview.png`, and
`__fullres.png`. The grade is predicted once from a reduced copy of the complete
frame, then applied unchanged to every native-resolution tile.

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
