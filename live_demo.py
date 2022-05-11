#!/usr/bin/env python
# coding: utf-8

# First, let's load the JSON file which describes the human pose task.  This is in COCO format, it is the category descriptor pulled from the annotations file.  We modify the COCO category slightly, to add a neck keypoint.  We will use this task description JSON to create a topology tensor, which is an intermediate data structure that describes the part linkages, as well as which channels in the part affinity field each linkage corresponds to.

# In[ ]:


import json
import trt_pose.coco
from multiprocessing import Process
import sys
sys.path.append("/opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps-1.0/apps/deepstream-test1-rtsp-out")
from eric3_final_class import*

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

#pipeline = Deepstream_Array_To_Rtsp()

# Next, we'll load our model.  Each model takes at least two parameters, *cmap_channels* and *paf_channels* corresponding to the number of heatmap channels
# and part affinity field channels.  The number of part affinity field channels is 2x the number of links, because each link has a channel corresponding to the
# x and y direction of the vector field for each link.

# In[ ]:


import trt_pose.models

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()


# Next, let's load the model weights.  You will need to download these according to the table in the README.

# In[ ]:


import torch

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'

model.load_state_dict(torch.load(MODEL_WEIGHTS))


# In order to optimize with TensorRT using the python library *torch2trt* we'll also need to create some example data.  The dimensions
# of this data should match the dimensions that the network was trained with.  Since we're using the resnet18 variant that was trained on
# an input resolution of 224x224, we set the width and height to these dimensions.

# In[ ]:


WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()


# Next, we'll use [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) to optimize the model.  We'll enable fp16_mode to allow optimizations to use reduced half precision.

# In[ ]:


import torch2trt

#model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)


# The optimized model may be saved so that we do not need to perform optimization again, we can just load the model.  Please note that TensorRT has device specific optimizations, so you can only use an optimized model on similar platforms.

# In[ ]:


OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

#torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)


# We could then load the saved model using *torch2trt* as follows.

# In[ ]:


from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


# We can benchmark the model in FPS with the following code

# In[ ]:


import time

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(10):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(10.0 / (t1 - t0))


# Next, let's define a function that will preprocess the image, which is originally in BGR8 / HWC format.

# In[ ]:


import cv2
import torchvision.transforms as transforms
import PIL.Image

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


# Next, we'll define two callable classes that will be used to parse the objects from the neural network, as well as draw the parsed objects on an image.

# In[ ]:


from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


# Assuming you're using NVIDIA Jetson, you can use the [jetcam](https://github.com/NVIDIA-AI-IOT/jetcam) package to create an easy to use camera that will produce images in BGR8/HWC format.
# 
# If you're not on Jetson, you may need to adapt the code below.

# In[ ]:


#from jetcam.usb_camera import USBCamera
# from jetcam.csi_camera import CSICamera
#from jetcam.utils import bgr8_to_jpeg

#camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30)
# camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)

#camera.running = True


# Next, we'll create a widget which will be used to display the camera feed with visualizations.

# In[ ]:


import ipywidgets
from IPython.display import display

image_w = ipywidgets.Image(format='jpeg')

display(image_w)


# Finally, we'll define the main execution loop.  This will perform the following steps
# 
# 1.  Preprocess the camera image
# 2.  Execute the neural network
# 3.  Parse the objects from the neural network output
# 4.  Draw the objects onto the camera image
# 5.  Convert the image to JPEG format and stream to the display widget

# In[ ]:
def drawpeak(topology,image, object_counts, objects, normalized_peaks):
    height = image.shape[0]
    width = image.shape[1]
    K = topology.shape[0]
    count = int(object_counts[0])
    for i in range(count):
        color1 = (0, 0, 255)
        color2 = (255, 0, 0)
        obj = objects[0][i]
        C = obj.shape[0]
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                peak = normalized_peaks[0][j][k]
                x = round(float(peak[1]) * width)
                y = round(float(peak[0]) * height)
                cv2.circle(image, (x, y), 3, color1, 2)

        '''for k in range(K):
            c_a = topology[k][2]
            c_b = topology[k][3]
            if obj[c_a] >= 0 and obj[c_b] >= 0:
                peak0 = normalized_peaks[0][c_a][obj[c_a]]
                peak1 = normalized_peaks[0][c_b][obj[c_b]]
                x0 = round(float(peak0[1]) * width)
                y0 = round(float(peak0[0]) * height)
                x1 = round(float(peak1[1]) * width)
                y1 = round(float(peak1[0]) * height)
                cv2.line(image, (x0, y0), (x1, y1), color2, 2)'''


## use video input
vid = cv2.VideoCapture('rtsp://admin:abc541287@192.168.0.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')

#vid = cv2.VideoCapture(0)
#vid_cnt=int(vid.get(7))
#def execute(change):
#Process(target=pipeline.rtspOut_streaming, daemon=True,args=[]).start()
def execute():
    #image = change['new']
    c0 = time.time()
    ret,image = vid.read()
    image=cv2.resize(image,(WIDTH,HEIGHT))
    c1 = time.time()
    t0 = time.time()
    data = preprocess(image)
    
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    t1 = time.time()
    a0 = time.time()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    drawpeak(topology,image, counts, objects, peaks)
    #draw_objects(image, counts, objects, peaks)
    a1 = time.time()
    #image_w.value = bgr8_to_jpeg(image[:, ::-1, :])
    #frame=cv2.resize(image,(image.shape[1]*3,image.shape[0]*3))
    frame=cv2.resize(image,(640,480))
    '''if (t1-t0)!=0:
        print(1/(t1-t0))'''
    cv2.imshow('image',frame)
    cv2.waitKey(1)
    b0 = time.time()
    pipeline.start_streaming(frame)
    b1 = time.time()
    print('model time : ' + str(t1-t0))
    print('draw time : ' + str(a1-a0))
    print('deepstream time:'+str(b1-b0))
    print('other time:'+str(c1-c0))
   
    

# If we call the cell below it will execute the function once on the current camera frame.

# In[ ]:

#while vid.isOpened():

for i in range(3000):
#while True:
    d0 = time.time()
    execute()
    d1 = time.time()
    print(d1-d0)
    if cv2.waitKey(1) == ord('q'):
        break

#graph_pipeline(pipeline, "/home/jetson2021030510/Downloads")
pipeline.end_streaming()



#execute({'new': camera.value})


# Call the cell below to attach the execution function to the camera's internal value.  This will cause the execute function to be called whenever a new camera frame is received.

# In[ ]:


#camera.observe(execute, names='value')


# Call the cell below to unattach the camera frame callbacks.

# In[ ]:


#camera.unobserve_all()

