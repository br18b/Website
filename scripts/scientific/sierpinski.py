import math
import random
import matplotlib.pyplot as plt

def regular_ngon(n, radius=1.0, rotation=None):
    if rotation is None:
        rotation = 0
    return [
        (
            radius * math.sin(rotation + 2 * math.pi * k / n),
            radius * math.cos(rotation + 2 * math.pi * k / n)
        )
        for k in range(n)
    ]

def sierpinski_ngon(n, start=None, n_points=200_000, step=None, weights=None):
    if n < 3:
        raise ValueError("n must be at least 3")

    points = regular_ngon(n)

    half_points = [
        ((x1 + x2) / 2, (y1 + y2) / 2)
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    ]

    vertices = points + half_points
    vertices = points

    if step is None:
        step = 1 / (2 + 2 * math.cos(2 * math.pi / n))

    if weights is None:
        weights = [1] * len(vertices)

    if len(weights) != len(vertices):
        raise ValueError("Length of weights must match the lenght of the vertices")
    if sum(weights) <= 0:
        raise ValueError("At least one weight must be positive")

    if start is None:
        start = points[0]
    x, y = start
    points = []

    for _ in range(n_points):
        vx, vy = random.choices(vertices, weights=weights, k=1)[0]
        x = step * x + (1 - step) * vx
        y = step * y + (1 - step) * vy
        points.append((x, y))

    return points

# Example: pentagon
n = 3
points = sierpinski_ngon(n, n_points=100_000, weights=[3,2,1])

xs, ys = zip(*points)

plt.figure(figsize=(6, 6))
plt.plot(xs, ys, ',', linestyle='None')
plt.axis("equal")
plt.axis("off")
plt.savefig("sierpinski_ngon.png", dpi=300, bbox_inches="tight", pad_inches=0)