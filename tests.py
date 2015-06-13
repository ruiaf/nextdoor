import unittest
import random
import numpy as np
from nextdoor import NearestNeighborsIndex

class SimpleTestCase(unittest.TestCase):
	def setUp(self):
		self.nn = NearestNeighborsIndex()
		self.vecs = [
			np.array([1.0, 1.0, 1.0 ,1.0]),
			np.array([1.0, 0.0, 1.0 ,1.0]),
			np.array([1.0, 1.0, 0.0 ,1.0]),
			np.array([1.0, 0.0, 0.0 ,1.0])]

		for i, vec in enumerate(self.vecs):
			self.nn[i] = vec
	
	def testAddedElements(self):
		assert len(self.nn) == 4
		for i, vec in enumerate(self.vecs):
			assert (self.nn[i] == vec).all()

	def testFindNeighbors(self):
		nearest = self.nn.knearest(self.vecs[3], 3)
		for i in range(1,3):
			assert(i in nearest)

	def testDictInterface(self):
		x = random.choice(self.nn.keys())
		assert (self.nn[x] == self.vecs[x]).all()

if __name__ == "__main__":
	unittest.main()
