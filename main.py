import kdtree
import numpy as np

point1 = (0, 1)
point2 = (2, 2)
point3 = [1, 1]

tree = kdtree.create(dimensions=2)

tree.add(point1)
tree.add(point2)
tree.add(point3)

# tree = tree.remove(point1)

# point, distance = tree.search_nn(point1)
# print(point)
# print(distance)

# tree.add(point1)

tree = tree.remove(point2)

point, distance = tree.search_nn(point2)
print(point.data[0])
# print(distance)

tree.add(point2)
