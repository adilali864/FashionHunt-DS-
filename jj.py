import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


from keras.models import load_model
loaded_model = load_model('my_model.h5')

import os
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

img_path = r"D:\images\image9.jpg"

img = image.load_img(img_path, target_size=(200, 200))
plt.imshow(img)
plt.show()
X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)
result = loaded_model.predict(X)
prediction = np.argmax(result) 
    
if prediction == 0:
    print("sho")
elif prediction == 1:
    print('t-shirt')
else:
    print('trouser')

