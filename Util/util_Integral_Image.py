import numpy as np
def get_integral_image(input_image):
    #print(input_image.shape)
    input_image = np.array(input_image)
    integral_image = np.zeros(input_image.shape)
    s = np.zeros(input_image.shape)
    for y in range(len(input_image)):
        #print(input_image[y].shape)
        for x in range (len(input_image[y])):
            s[y][x] = s[y-1][x] + input_image[y][x] if y-1 >= 0 else input_image[y][x]
            integral_image[y][x] = integral_image[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return integral_image