import numpy as np
from PIL import Image
import sys

def convert_image(image_path, output_path):
    #converts the image to grayscale after opening
    try:
        img = Image.open(image_path).convert("L")
    except FileNotFoundError:
        print(f"file not found, fix the file path")
        return
    #resizes the image to  664 x 64
    reimg = img.resize((64,64), resample = Image.Resampling.NEAREST)
    #converts the image into a Numpy array
    pixelate = np.array(reimg)
    #[cite_start]
    #checks where the walls are and sets the Start and goal
    maze_grid = np.where(pixelate < 128, 1, 0)
    maze_grid[1, 1] = 2
    maze_grid[62, 62] = 3
    #saves as a text file so it can be loaded by the agent
    np.savetxt(output_path, maze_grid, fmt='%d', delimiter=' ')
    print(f"Success! Converted '{image_path}' to '{output_path}'.")
    print(f"Grid Shape: {maze_grid.shape}")
    print("You can now load this file in your main agent code.")
#test
convert_image("MAZE_0.png", "training.txt")