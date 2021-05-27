import pandas as pd
from scipy.spatial import distance_matrix


def generate_distance_matrix(points):
  return distance_matrix(points)


points = [
  [4, 3],
  [5, 8],
  [5, 7],
  [9, 3],
  [11, 6],
  [13, 8]
]

print(generate_distance_matrix(points))
