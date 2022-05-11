import argparse
import sys
sys.path.append('../')

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
import numpy as np
import pyds
import cv2
import time
import multiprocessing as mp
import os

def ndarray_to_gst_buffer(array:np.ndarray)->Gst.Buffer:
    return Gst.Buffer.new_wrapped(array.tobytes())


def gst_rtspgo():

    print("sub line")
    # create an event loop and feed gstreamer bus mesages to it

    loop = GObject.MainLoop()
    loop.run()

# 在使用前記得先安裝graphviz接著在command輸入export GST_DEBUG_DUMP_DOT_DIR=/tmp/
def graph_pipeline(pipeline, graph_save_location):

    Gst.debug_bin_to_dot_file(pipeline,Gst.DebugGraphDetails.ALL,"pipeline")
    path = graph_save_location + "/" + "pipeline.png"    
    os.system(f"dot -Tpng -o {path} /tmp/pipeline.dot")
  

def main():
    # Standard GStreamer initialization
    
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")

    pipeline = Gst.Pipeline()
  
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    
    appsource = Gst.ElementFactory.make("appsrc", "numpy-source")
    if not appsource:
        sys.stderr.write(" Unable to create appsource \n")

    

    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert","nv-videoconv")
    if not nvvideoconvert:
        sys.stderr.write(" Unable to create nvvid1")

    caps1 = Gst.ElementFactory.make("capsfilter","capsfilter1")
    if not caps1:
        sys.stderr.write(" Unable to create capsf1")
 
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
      
    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    
    # Create a caps filter
    caps2 = Gst.ElementFactory.make("capsfilter", "filter")
    if not caps2:
        sys.stderr.write(" Unable to create capsf2")
    
    # Make the encoder
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
   
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write(" Unable to create h264 parser \n")
    
    # Make the payload-encode video into RTP packets
    rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")
    
    # Make the UDP sink    
    udpsink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not udpsink:
        sys.stderr.write(" Unable to create udpsink")



    # Set up the property
    caps_in = Gst.Caps.from_string("video/x-raw,format=RGBA,width=640,height=480,framerate=30/1")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12,width=640,height=480,framerate=30/1")
    appsource.set_property('caps', caps_in)
    #appsource.set_property('um-buffers',5)
    caps1.set_property('caps',caps)   
    streammux.set_property('width', 640)
    streammux.set_property('height', 480)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    caps2.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    encoder.set_property('bitrate', 4000000)
    if is_aarch64():
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        encoder.set_property('bufapi-version', 1)
    updsink_port_num = 5400
    udpsink.set_property('host', '127.0.0.1')
    udpsink.set_property('port', updsink_port_num)
    udpsink.set_property('async', False)
    udpsink.set_property('sync', 1)
 

   
    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")

 

    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)



    print("Adding elements to Pipeline \n")

    pipeline.add(appsource)
    pipeline.add(nvvideoconvert)
    pipeline.add(caps1)
    pipeline.add(streammux)
    pipeline.add(nvvidconv)
    pipeline.add(caps2)
    pipeline.add(encoder)
    pipeline.add(h264parser)
    pipeline.add(rtppay)
    pipeline.add(udpsink)

    print("Linking elements in the Pipeline \n")

    appsource.link(nvvideoconvert)
    nvvideoconvert.link(caps1)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
    srcpad = caps1.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")

     
    srcpad.link(sinkpad)

    streammux.link(nvvidconv)    
    #queue1.link(nvvidconv) 
    nvvidconv.link(caps2)
    #queue2.link(caps2)
    caps2.link(encoder)
    #queue3.link(encoder)
    encoder.link(h264parser)
    #queue4.link(h264parser)
    h264parser.link(rtppay)
    #rtppay.link(queue5)
    #queue5.link(udpsink)
    rtppay.link(udpsink)



    # Start streaming
    rtsp_port_num = 8554
    
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)
    
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, "H264"))
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)
    
    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)
    
    
    # start play back and listen to events
    print("Starting pipeline \n")	
    pipeline.set_state(Gst.State.PLAYING)

    mp.Process(target=gst_rtspgo,daemon = True,args = []).start()
 
    
    for _ in range(10):

        send1 = time.time()
        print("frame:", _)
        

        if _ <= 500:
            arr=np.zeros((480,640))
            

        elif _ == 999:
            arr=np.zeros((480,640))

        else:
            arr=np.random.randint(low=0,high=255,size=(480,640,3),dtype=np.uint8)
            arr=cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
            
        
        appsource.emit("push-buffer",ndarray_to_gst_buffer(arr))
        
        time.sleep(0.1)

        send2 = time.time()
        print("sending time:",send2-send1)
        print(" ")

    #print(appsource.list_properties())
    appsource.emit("end-of-stream")
    
    graph_pipeline(pipeline, "/home/jetson2021030510/Downloads")
    # cleanup
    print("Exiting app\n")
    
    
 
    pipeline.set_state(Gst.State.NULL)

    


if __name__ == '__main__':

    main()





