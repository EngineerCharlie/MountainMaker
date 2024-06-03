import numpy as np

def Project_PCD_TO_MAP(pcdToProject, resolution=5):
    point_cloud = np.asarray(pcdToProject.points)
    #orig_colors = np.asarray(pcdToProject.colors)

    # Define heightmap resolution (adjust as needed)
    #heightmap_resolution = 5 # Example: 0.1 units per cell

    # Calculate grid dimensions
    x_min, y_min, z_min = np.min(point_cloud[:, :3], axis=0)
    x_max, y_max, z_max = np.max(point_cloud[:, :3], axis=0)

    #if X dim is larger the heightmap should be X*X
    #if Z dim is larger the heightmap should be Z*Z
    x_range = x_max-x_min
    z_range = z_max-z_min
    if(x_range > z_range):
        num_cols = int(x_range / resolution) + 1
    else:
        num_cols = int(z_range / resolution) + 1

    #print("Y range: ", (y_max-y_min))

    # Ensure a square heightmap by adjusting the number of rows
    num_rows = num_cols

    # Initialize heightmap
    heightmap = np.full((num_rows, num_cols), 0)

    #colormap = np.full((num_rows, num_cols, 3), [0,0,0])

    # Project z-values onto the heightmap
    for i in range(len(point_cloud)):
        x, y, z = point_cloud[i]
        #row = int((y - y_min) / resolution)
        row = int((z - z_min) / resolution)
        col = int((x - x_min) / resolution)
        heightmap[row, col] = y
        #colormap[row, col] = orig_colors[i]

    return heightmap
