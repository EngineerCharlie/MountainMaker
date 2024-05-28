from PaintApp.pypaint import Paint

from GAN.GenerateMountain import generate_mountain_from_file

c = Paint()
c.run()
generate_mountain_from_file("my_mountain.png", True)
