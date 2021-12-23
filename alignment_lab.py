import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
training_data  =  np.load("np_data/run_input_training_data.npy", allow_pickle=True)
t_0            = np.load("saved_while_training/t_data0.npy", allow_pickle=True)
t_1            = np.load("saved_while_training/t_data1.npy", allow_pickle=True)
#st.write("This is the training set")
#image_index = st.text_input("value of image index", 0)
y           = st.slider('Change image',min_value = 0,max_value = len(training_data))

s_hat       = training_data[y][1]
s_delta     = training_data[y][2]
features    = training_data[y][4]
intensities = training_data[y][3]
image       = training_data[y][0]

s_hat_0       = t_0[y][1]
features_0    = t_0[y][4]
image_0       = t_0[y][0]

s_hat_1       = t_1[y][1]
features_1    = t_1[y][4]
image_1       = t_1[y][0]





#bb          = training_data[y][5]
#bb_center_x = bb[0]
#bb_center_y = bb[1]
#bb_w        = bb[2]
#bb_h        = bb[3]
fig, ax     = plt.subplots(2,2)

#ax[0][0].add_patch(Rectangle((bb_center_x,bb_center_y), bb_w, bb_h, ec="orange", fill=None, alpha=1))
#ax[0][1].add_patch(Rectangle((bb_center_x,bb_center_y), bb_w, bb_h, ec="orange", fill=None, alpha=1))


#T=0
ax[0][0].scatter(s_hat[:,0], s_hat[:,1], color="red", s=1)
ax[0][0].imshow(image, cmap="gray")

#ax[0][0].scatter(features[:,0], features[:,1], color="green", s=1)
#ax[1].scatter(features[:,0], features[:,1], color="green", s=1)
ax[0][1].scatter(s_hat[:,0] + s_delta[:,0], s_hat[:,1] + s_delta[:,1], color="red", s=1)
ax[0][1].imshow(image, cmap="gray")

#T=1
ax[1][0].scatter(s_hat_0[:,0], s_hat_0[:,1], color="red", s=1)
ax[1][0].imshow(image_0, cmap="gray")
#ax[1][0].scatter(features_0[:,0], features_0[:,1], color="green", s=1)

#T=2
ax[1][1].scatter(s_hat_1[:,0], s_hat_1[:,1], color="red", s=1)
ax[1][1].imshow(image_1, cmap="gray")
#ax[1][1].scatter(features_1[:,0], features_1[:,1], color="green", s=1)


st.pyplot(fig)

#st.image(image)

