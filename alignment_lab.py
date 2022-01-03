import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

st.set_page_config(layout="wide")

@st.cache
def load_data():
    training_data  =  np.load("np_data/run_input_training_data.npy", allow_pickle=True)
    t_0            = np.load("saved_while_training/t_data0.npy", allow_pickle=True)
    t_1            = np.load("saved_while_training/t_data1.npy", allow_pickle=True)
    t_2            = np.load("saved_while_training/t_data2.npy", allow_pickle=True)
    t_3            = np.load("saved_while_training/t_data3.npy", allow_pickle=True)
    t_4            = np.load("saved_while_training/t_data4.npy", allow_pickle=True)
    t_5            = np.load("saved_while_training/t_data5.npy", allow_pickle=True)
    t_6            = np.load("saved_while_training/t_data6.npy", allow_pickle=True)
    t_7            = np.load("saved_while_training/t_data7.npy", allow_pickle=True)
    t_8            = np.load("saved_while_training/t_data8.npy", allow_pickle=True)
    t_9            = np.load("saved_while_training/t_data9.npy", allow_pickle=True)


    return training_data,t_0,t_1,t_2,t_3,t_4,t_5,t_6,t_7,t_8,t_9
#st.write("This is the training set")
#image_index = st.text_input("value of image index", 0)

training_data,t_0,t_1,t_2,t_3,t_4,t_5,t_6,t_7,t_8,t_9 = load_data()



y           = st.slider('Change image',min_value = 0,max_value = len(training_data))




def plot_step(row, column, datafile,ax):
    s_hat       = datafile[y][1]
    #features    = datafile[y][4]
    image       = datafile[y][0]

    ax[row][column].scatter(s_hat[:,0], s_hat[:,1], color="red", s=0.4)
    ax[row][column].imshow(image, cmap="gray")



#bb          = training_data[y][5]
#bb_center_x = bb[0]
#bb_center_y = bb[1]
#bb_w        = bb[2]
#bb_h        = bb[3]
fig, ax     = plt.subplots(2,5)
fig.set_size_inches(36, 20)

#ax[0][0].add_patch(Rectangle((bb_center_x,bb_center_y), bb_w, bb_h, ec="orange", fill=None, alpha=1))
#ax[0][1].add_patch(Rectangle((bb_center_x,bb_center_y), bb_w, bb_h, ec="orange", fill=None, alpha=1))


plot_step(0,0,training_data,ax)
plot_step(0,1,t_0,ax)
plot_step(0,2,t_1,ax)
plot_step(0,3,t_2,ax)
plot_step(0,4,t_3,ax)
plot_step(1,0,t_4,ax)
plot_step(1,1,t_5,ax)
plot_step(1,2,t_6,ax)
plot_step(1,3,t_7,ax)
plot_step(1,4,t_8,ax)






#ax[0][0].scatter(features[:,0], features[:,1], color="green", s=1)
#ax[1].scatter(features[:,0], features[:,1], color="green", s=1)
#ax[0][1].scatter(s_hat[:,0] + s_delta[:,0], s_hat[:,1] + s_delta[:,1], color="red", s=1)
#ax[0][1].imshow(image, cmap="gray")



st.pyplot(fig)

#st.image(image)
