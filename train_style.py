"""Train a content-invariant movie-look encoder from the indexed film corpus."""

import argparse
import random

import torch
from torch import nn
import torch.nn.functional as F

from colorgrade.media import load_display_image
from colorgrade.corpus import load_or_build_index, prepare_image
from colorgrade.style import MovieStyleEncoder, supervised_contrastive_loss


def augment(image: torch.Tensor) -> torch.Tensor:
    # Geometry may change; colour does not, because colour is the supervision.
    if random.random() < 0.5:
        image = image.flip(-1)
    scale = random.uniform(0.78, 1.0)
    size = image.shape[-1]
    crop = max(32, round(size * scale))
    top = random.randrange(size - crop + 1)
    left = random.randrange(size - crop + 1)
    return F.interpolate(image[:, top:top+crop, left:left+crop][None],
                         (size, size), mode="bilinear", align_corners=False)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="film_index.pt")
    parser.add_argument("--output", default="movie_style_encoder.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--movies-per-batch", type=int, default=8)
    parser.add_argument("--frames-per-movie", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
        ("cuda" if torch.cuda.is_available() else "cpu")
    index = load_or_build_index(args.index, device=device)
    grouped = {}
    for path, movie in zip(index["paths"], index["movies"]):
        grouped.setdefault(movie, []).append(path)
    grouped = {movie: paths for movie, paths in grouped.items()
               if len(paths) >= args.frames_per_movie}
    movies = sorted(grouped)
    movie_id = {movie: i for i, movie in enumerate(movies)}
    print(f"training style encoder on {len(movies)} movies", flush=True)

    model = MovieStyleEncoder().to(device).train()
    classifier = nn.Linear(model.embedding_size, len(movies)).to(device)
    optimizer = torch.optim.AdamW([*model.parameters(), *classifier.parameters()],
                                  lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        accuracy = 0.0
        for _ in range(args.steps_per_epoch):
            chosen_movies = random.sample(movies, min(args.movies_per_batch, len(movies)))
            images, labels = [], []
            for movie in chosen_movies:
                for path in random.sample(grouped[movie], args.frames_per_movie):
                    image = prepare_image(load_display_image(path, size=None))
                    images.append(augment(image))
                    labels.append(movie_id[movie])
            batch = torch.stack(images).to(device)
            labels = torch.tensor(labels, device=device)
            embedding = model(batch, destroy_content=True)
            logits = classifier(embedding)
            contrastive = supervised_contrastive_loss(embedding, labels)
            classification = F.cross_entropy(logits, labels)
            loss = contrastive + 0.25 * classification
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            accuracy += (logits.argmax(1) == labels).float().mean().item()
        print(f"epoch {epoch:3d} loss {running/args.steps_per_epoch:.4f} "
              f"movie_acc {accuracy/args.steps_per_epoch:.3f}", flush=True)

    torch.save({"style_encoder_version": 1, "state_dict": model.eval().state_dict(),
                "movies": movies, "epochs": args.epochs}, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
