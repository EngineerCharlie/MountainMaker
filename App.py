from PaintApp.pypaint import Paint

from GAN.GenerateMountain import generate_mountain_from_file

from ModelGen3D.Main import generate_3d_model

c = Paint()
c.run()
generate_mountain_from_file("my_mountain.png", True)
generate_3d_model()
