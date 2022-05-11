
# coding: utf-8

# In[5]:


import cv2
import numpy as np
import time

# In[7]:


#cap = cv2.VideoCapture(0) ##非常非常重要 羅技相機 預設擷取為CAP_MSMF只能使用640*480不能自由更改解析度(顯示有問題) 改為CAP_DSHOW 可設定相機內既有的所有解析度
#cap = cv2.VideoCapture()
#cap1 = cv2.VideoCapture('rtsp://admin:abc541287@192.168.0.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')

cap1=cv2.VideoCapture('rtsp://localhost:8554/ds-test')



count = 0
while cap1.isOpened():  
    receive1 = time.time()
    #ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if (ret1)==True: 

	
        cv2.imshow('frame1',frame1)
        print("frame:",count)
        count = count+1
        receive2 = time.time()
        print("receive time:",receive2-receive1)

        #if cv2.waitKey(1) & 0xFF == 27:
        if cv2.waitKey(1) == ord('q'):
            break
        
    else:
        break

        
cap1.release()
cv2.destroyAllWindows()




