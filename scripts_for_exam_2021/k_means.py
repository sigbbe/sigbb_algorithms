from math import sqrt, inf

"""
Calculate the manhattan distance.

@a: first coordinate
@b: second coordinate

"""
def manhattan(a, b):
  return abs(a[0] - b[0]) + abs(a[1] - b[1])

"""
Calculate the eucledian distance.

@a: first coordinate
@b: second coordinate
@root: Boolean defaulted to True telling wether or not to take the root of the distance.
"""
def eucledian(a, b, root = True):
    if root:
      return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    return (a[0] - b[0])**2 + (a[1] - b[1])**2

"""

Determines the closest centroid from a given point, using Manhattan-distance (abs(x1 - x2) + abs(y1- y2))

@calc: function for determining distance calculation
@centroids: list of given predetermined centroids
@point: point to calculate the distance from

"""
def determine_closest_centroid(calc, centroids, point):
  closestPoint = ['N/A', inf]

  for key, value in centroids.items():
    distance = manhattan(value, point) if calc == "manhattan" else eucledian(value, point, False)

    if closestPoint[1] > distance:
      closestPoint[0] = key
      closestPoint[1] = distance

  return closestPoint

centroids = {
  'M1': [2.0, 0.0],
  'M2': [3.0, 4,0],
}

print(determine_closest_centroid("eucledian", centroids, (1, 4)))
