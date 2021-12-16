import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
training_data  =  np.load("train_1_quadruplets_with_bb.npy", allow_pickle=True)


st.write("This is the training set")


y           = st.slider('Change image',min_value = 0,max_value = len(training_data))

s_hat       = training_data[y][1]
s_delta     = training_data[y][2]
features    = training_data[y][4]
intensities = training_data[y][3]
image       = training_data[y][0]
bb          = training_data[y][5]
bb_center_x = bb[0][0]
bb_center_y = bb[0][1]
bb_w        = bb[0][2]
bb_h        = bb[0][3]
fig, ax     = plt.subplots(1,2)

ax[0].add_patch(Rectangle((bb_center_x,bb_center_y), bb_w, bb_h, ec="orange", fill=None, alpha=1))
ax[1].add_patch(Rectangle((bb_center_x,bb_center_y), bb_w, bb_h, ec="orange", fill=None, alpha=1))

ax[0].scatter(s_hat[:,0], s_hat[:,1], color="red", s=1)
ax[0].imshow(image, cmap="gray")

ax[0].scatter(features[:,0], features[:,1], color="green", s=1)
#ax[1].scatter(features[:,0], features[:,1], color="green", s=1)
ax[1].scatter(s_hat[:,0] + s_delta[:,0], s_hat[:,1] + s_delta[:,1], color="red", s=1)
ax[1].imshow(image, cmap="gray")


st.pyplot(fig)

#st.image(image)

