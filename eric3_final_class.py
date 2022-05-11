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


class Deepstream_Array_To_Rtsp:

    def __init__(self):

        # Standard GStreamer initialization
        GObject.threads_init()
        Gst.init(None)


        #------------------------- part1 create element ---------------------------------------------------------#


        print("Creating Pipeline \n ")
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            sys.stderr.write(" Unable to create Pipeline \n")

        self.appsource = Gst.ElementFactory.make("appsrc", "numpy-source")
        if not self.appsource:
            sys.stderr.write(" Unable to create appsource \n")

        self.nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nv-videoconv")
        if not self.nvvideoconvert:
            sys.stderr.write(" Unable to create nvvid1")

        self.caps1 = Gst.ElementFactory.make("capsfilter", "capsfilter1")
        if not self.caps1:
            sys.stderr.write(" Unable to create capsf1")

        # Create nvstreammux instance to form batches from one or more sources.
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not self.streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")

        # Use convertor to convert from NV12 to RGBA as required by nvosd
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not self.nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv \n")

        # Create a caps filter
        self.caps2 = Gst.ElementFactory.make("capsfilter", "filter")
        if not self.caps2:
            sys.stderr.write(" Unable to create capsf2")

        # Make the encoder
        self.encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        if not self.encoder:
            sys.stderr.write(" Unable to create encoder")

        self.h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        if not self.h264parser:
            sys.stderr.write(" Unable to create h264 parser \n")

        # Make the payload-encode video into RTP packets
        self.rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        if not self.rtppay:
            sys.stderr.write(" Unable to create rtppay")

        # Make the UDP sink
        self.udpsink = Gst.ElementFactory.make("udpsink", "udpsink")
        if not self.udpsink:
            sys.stderr.write(" Unable to create udpsink")


        #----------------------------- part2 Set up the property -----------------------------------------------#


        self.caps_in = Gst.Caps.from_string("video/x-raw,format=RGBA,width=640,height=480,framerate=30/1")
        self.caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12,width=640,height=480,framerate=30/1")
        self.appsource.set_property('caps', self.caps_in)
        self.caps1.set_property('caps', self.caps)
        self.streammux.set_property('width', 640)
        self.streammux.set_property('height', 480)
        self.streammux.set_property('batch-size', 1)
        self.streammux.set_property('batched-push-timeout', 4000000)
        self.caps2.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
        self.encoder.set_property('bitrate', 4000000)
        if is_aarch64():
            self.encoder.set_property('preset-level', 1)
            self.encoder.set_property('insert-sps-pps', 1)
            self.encoder.set_property('bufapi-version', 1)
        self.updsink_port_num = 5400
        self.udpsink.set_property('host', '127.0.0.1')
        self.udpsink.set_property('port', self.updsink_port_num)
        self.udpsink.set_property('async', False)
        self.udpsink.set_property('sync', 1)


        #------------------------------ part3 add all element into pipeline -----------------------------------#


        print("Adding elements to Pipeline \n")
        self.pipeline.add(self.appsource)
        self.pipeline.add(self.nvvideoconvert)
        self.pipeline.add(self.caps1)
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.caps2)
        self.pipeline.add(self.encoder)
        self.pipeline.add(self.h264parser)
        self.pipeline.add(self.rtppay)
        self.pipeline.add(self.udpsink)


        #------------------------------ part4 link all needed element in the pipeline --------------------------#


        print("Linking elements in the Pipeline \n")
        self.appsource.link(self.nvvideoconvert)
        self.nvvideoconvert.link(self.caps1)

        self.sinkpad = self.streammux.get_request_pad("sink_0")
        if not self.sinkpad:
            sys.stderr.write(" Unable to get the sink pad of streammux \n")

        self.srcpad = self.caps1.get_static_pad("src")
        if not self.srcpad:
            sys.stderr.write(" Unable to get source pad of decoder \n")

        self.srcpad.link(self.sinkpad)

        self.streammux.link(self.nvvidconv)

        self.nvvidconv.link(self.caps2)

        self.caps2.link(self.encoder)

        self.encoder.link(self.h264parser)

        self.h264parser.link(self.rtppay)

        self.rtppay.link(self.udpsink)

        # create an event loop and feed gstreamer bus mesages to it
        self.loop = GObject.MainLoop()
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", bus_call, self.loop)

        # Start streaming
        self.rtsp_port_num = 8554

        self.server = GstRtspServer.RTSPServer.new()
        self.server.props.service = "%d" % self.rtsp_port_num
        self.server.attach(None)

        self.factory = GstRtspServer.RTSPMediaFactory.new()
        self.factory.set_launch(
            "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (
            self.updsink_port_num, "H264"))
        self.factory.set_shared(True)
        self.server.get_mount_points().add_factory("/ds-test", self.factory)

        print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % self.rtsp_port_num)

        # start play back and listen to events
        print("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)


    def ndarray_to_gst_buffer(self, array: np.ndarray) -> Gst.Buffer:

        return Gst.Buffer.new_wrapped(array.tobytes())


    def start_streaming(self,input_array):  
      
        input_array = cv2.cvtColor(input_array, cv2.COLOR_BGR2RGBA)
        convert = self.ndarray_to_gst_buffer(input_array)
        self.appsource.emit("push-buffer", convert)


    def rtspOut_streaming(self):

        try:
            self.loop.run()
        except:
            pass


    def end_streaming(self):

        self.appsource.emit("end-of-stream")
        #cleanup
        print("Exiting app\n")
        self.pipeline.set_state(Gst.State.NULL)

    # 在使用前記得先安裝graphviz接著在command輸入export GST_DEBUG_DUMP_DOT_DIR=/tmp/
    def graph_pipeline(self, pipeline, graph_save_location):

        Gst.debug_bin_to_dot_file(pipeline,Gst.DebugGraphDetails.ALL,"pipeline")
        path = graph_save_location + "/" + "pipeline.png"   
        os.system(f"dot -Tpng -o {path} /tmp/pipeline.dot")
    

#---------------------------------------- for testing code ------------------------------------------------------#


'''if __name__ == '__main__':

    pipeline = Deepstream_Array_To_Rtsp()

    for _ in range(500):

        image = np.random.randint(low=0, high=255, size=(480, 640, 3), dtype=np.uint8)       
        pipeline.start_streaming(image)      
        time.sleep(0.01)

    pipeline.end_streaming()'''







