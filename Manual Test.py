import numpy as np
from luminoth import Detector, read_image, vis_objects
from matplotlib import pyplot as plt

image = read_image('images/image1.jpg')

# If no checkpoint specified, will assume `accurate` by default.
# The Detector can also take a config object.

detector = Detector(checkpoint='fast')

# Returns a dictionary with the detections.
objects = detector.predict(image)

print(objects)

output = vis_objects(image, objects)

plt.subplot(1, 2, 1), plt.imshow(image), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(np.asarray(output)), plt.title('Output')
plt.show()

output.save('fast-out.png')
