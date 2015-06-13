import numpy as np
import heapq
from scipy.spatial import distance

class NearestNeighborsIndex(object):
	def __init__(self):
		self.index = {}

	def __getitem__(self, key):
		return self.index[key]

	def __setitem__(self, key, value):
	        self.index[key] = value

	def __len__(self):
	        return len(self.index)

	def __iter__(self):
	        return iter(self.index)

	def keys(self):
		return self.index.keys()

	def knearest(self, value, k=10):
		distances = {}
		for key, other_value in self.index.iteritems():
			distances[key] = distance.euclidean(value, other_value)
		nearest = heapq.nsmallest(k, distances, key=distances.get)
		return nearest
