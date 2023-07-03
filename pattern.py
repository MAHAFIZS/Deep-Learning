import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution need to be  divisible by 2 * tile size.")
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        rep = self.resolution // (2 * self.tile_size)
        self.output = np.concatenate((np.ones((self.tile_size, self.tile_size)), np.zeros((self.tile_size, self.tile_size))), axis = 1)
        temp_out = np.concatenate((np.zeros((self.tile_size, self.tile_size)), np.ones((self.tile_size, self.tile_size))), axis = 1)
        self.output = np.tile(self.output, rep)
        temp_output = np.tile(temp_out, rep)
        self.output = np.vstack((temp_output, self.output))
        self.output = np.tile(self.output, (rep, 1))
        return self.output.copy()
    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='gray')
        plt.show()
class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None
    def draw(self):
        x_axis = np.linspace(0, self.resolution-1, self.resolution)
        y_axis = np.linspace(0, self.resolution-1, self.resolution)
        x, y = np.meshgrid(x_axis, y_axis)
        self.output = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
        self.output = self.output < self.radius
        return self.output.copy()
    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        a = np.linspace(0, 1, self.resolution)
        b = np.linspace(0, 1, self.resolution)
        a, b = np.meshgrid(a, b)
        self.output = np.dstack((a, b, np.fliplr(a)))
        return self.output.copy()

    def show(self):
        plt.imshow(self.draw())
        plt.show()