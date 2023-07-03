from pattern import Checker
from pattern import Circle
from pattern import Spectrum
from generator import ImageGenerator


# Checker
checker=Checker(100,10)
checker.show()

# Circle
circle=Circle(100,25,(50,50))
circle.show()

# Color Spectrum
color_spectrum=Spectrum(100)
color_spectrum.show()

# Image Generator
Img_gen= ImageGenerator('./data/exercise_data/', './data/Labels.json', 12, [32, 32, 3])
Img_gen.show()
