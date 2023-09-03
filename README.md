# colorspace

![color wheel](image.jpeg)

> Assign color hues to a collection of text fragments based on embeddings.

Embeddings are high-dimensional, but can be compared to one another using
cosine similarity, which measures the angle between them. Interestingly,
points on a colorwheel are similarly separated by angle.

This implementation of embedding coloring works by brute-force: cosine
similarity of embeddings is compared against candidate colorspace arrangements
to optimize for the correlation between the two. This way, proximity in
colorspace corresponds to proximity in embedding space as closely as possible.

The limitation of this methodology is that brute-force puts a cap on the number
of embeddings that can be optimally arranged. However, **clustering** ensures
that a larger number of text embeddings can still be semantically arranged. And
for the sake of interpretability, **clusters** will often be what the human eye
picks up on anyway.

## Usage

To run coloring on an example set of words, run:

```sh
poetry install
poetry run python -m colorspace
```
