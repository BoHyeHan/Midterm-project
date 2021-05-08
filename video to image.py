#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[2]:


from cv2 import cv2 

vidcap = cv2.VideoCapture('Woman.mp4')

count = 0

while(vidcap.isOpened()):
    ret, image = vidcap.read()
    
    cv2.imwrite("image_capture/women%d.jpg" %count, image)
    
    print('Saved frame %d.jpg' % count)
    count += 1
    
vidcap.release()


# In[ ]:




