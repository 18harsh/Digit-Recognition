import pygame , random
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns



pygame.init()
win = pygame.display.set_mode((300,300))
pygame.display.set_caption("Paint")

model = keras.models.load_model('saved_models/digits-25-trained-model.h5')
class_names = ["0","1","2","3","4","5","6","7","8","9"]
last_pos = (0,0)
radius = 10
draw_on = False

def roundline(win, color , start, end, radius):
    dx =  end[0] - start[0]
    dy =  end[1] - start[1]

    distance = max(abs(dx),abs(dy))
    for i in range(distance):
        x = int(start[0]+float(i)/distance*dx)
        y = int(start[1]+float(i)/distance*dy)
        pygame.draw.circle(win,color,(x,y),radius)

try:
    while True:
        event = pygame.event.wait()     
        if event.type == pygame.QUIT:
            raise StopIteration
        if event.type == pygame.MOUSEBUTTONDOWN:
            color = (random.randrange(256),random.randrange(256),random.randrange(256))
            pygame.draw.circle(win,color,event.pos,radius)  
            draw_on = True
        if event.type == pygame.MOUSEBUTTONUP:
            draw_on = False
        if event.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(win,color,event.pos,radius)
                roundline(win, color , event.pos,last_pos , radius)         
            last_pos = event.pos
        pygame.display.flip()
except StopIteration:
    pass


surf_array = pygame.surfarray.array3d(win) 
surf_array = np.rot90(surf_array, k=1, axes=(1, 0))
surf_array = np.flip(surf_array, 1)

# print(surf_array)

import pickle
pickle_out = open("num.pickle","wb")
pickle.dump(surf_array,pickle_out)
pickle_out.close()

def get_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img,(25,25))

X = get_image(surf_array)

IMG_SIZE = 25
new_array = cv2.resize(X,(IMG_SIZE,IMG_SIZE))
new_array=X.reshape(-1,25,25,1)
prediction = model.predict(new_array)
plt.grid(False)
new_array = cv2.resize(X,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array,cmap=plt.cm.gray)

plt.title("prediction "+class_names[np.argmax(prediction[0])])
plt.show()

#final plot
def plot_image(prediction, img):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction)
    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
               100*np.max(prediction),
               ),
                color="blue")
def plot_value_array(prediction):
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction, color="#888888")
    plt.ylim([0,1])
    predicted_label = np.argmax(prediction)
    thisplot[predicted_label].set_color('blue')

plt.figure(figsize=(8,12))
plt.subplot(5, 2, 1)
plot_image(prediction[0], new_array)
# bar chart
plt.subplot(5, 2, 2)
plot_value_array(prediction[0])
plt.show()  

pygame.quit()

  