import os
import threading
import time
import subprocess
import signal

# GStreamer / GObject
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

# Flask for REST API
from flask import Flask, request, jsonify

# MQTT (optional)
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# OpenCV + YOLO (optional for object detection)
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

###############################################################################
# 1. INITIALIZATION
###############################################################################

# Initialize GStreamer
Gst.init(None)

app = Flask(__name__)

# Global references
rtsp_server = None           # Will hold the RTSP server instance
main_loop = None             # GLib MainLoop
pipeline_lock = threading.Lock()
object_detection_enabled = False  # Toggle for YOLO detection

# Vide file path relative to server.py
VIDEO_FILE_PATH = "sample_video.mp4"

# MQTT global references
mqtt_client = None
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_CONTROL = "stream/control"
MQTT_TOPIC_STATUS  = "stream/status"

###############################################################################
# 2. OPTIONAL: OBJECT DETECTION WITH OPENCV + YOLO
###############################################################################

YOLO_CONFIG_PATH  = "yolov3.cfg"     # Adjust path
YOLO_WEIGHTS_PATH = "yolov3.weights" # Adjust path
YOLO_NAMES_PATH   = None             # Optional: a .names file for class labels

net = None
output_layer_names = None
class_names = []

def load_yolo_model():
    global net, output_layer_names, class_names
    
    if not (YOLO_CONFIG_PATH and YOLO_WEIGHTS_PATH):
        print("[WARNING] YOLO config or weights not provided.")
        return
    
    print("[INFO] Loading YOLO model...")
    net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Load class names if provided
    if YOLO_NAMES_PATH and os.path.exists(YOLO_NAMES_PATH):
        with open(YOLO_NAMES_PATH, 'r') as f:
            class_names = [cname.strip() for cname in f.readlines()]
    else:
        class_names = []

###############################################################################
# 3. GSTREAMER RTSP SERVER
###############################################################################

class CustomRtspMediaFactory(GstRtspServer.RTSPMediaFactory):
    """
    A custom MediaFactory, creation decoupled so YOLO can be added later on
    """
    def __init__(self, video_file, **properties):
        super(CustomRtspMediaFactory, self).__init__(**properties)
        self.video_file = video_file
        self.launch_string = self.build_launch_string()
        self.set_launch(f"( {self.launch_string} )")

    def build_launch_string(self):
        # Replace the source element with a video file (filesrc)
        # Make sure to decode the video file using a suitable decoder like decodebin.
        if object_detection_enabled and OPENCV_AVAILABLE:
            # Use appsink with video file for YOLO detection
            launch_desc = (
                f"filesrc location={self.video_file} ! decodebin ! videoconvert ! queue ! video/x-raw,format=BGR ! "
                "tee name=t "
                "t. ! queue ! appsink name=mysink emit-signals=true sync=false "
                "appsrc name=mysrc is-live=true format=3 ! queue ! videoconvert ! "
                "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay name=pay0 pt=96"
            )
        else:
            # Simple pipeline: filesrc -> decodebin -> x264enc -> rtph264pay
            launch_desc = (
                f"filesrc location={self.video_file} ! decodebin ! videoconvert ! "
                "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay name=pay0 pt=96"
            )
        
        return launch_desc

###############################################################################
# 4. APPSINK CALLBACK FOR OBJECT DETECTION
###############################################################################
appsrc_global = None  # Will be set during do_configure

def on_new_sample(sink, pipeline):

    global net, output_layer_names, appsrc_global
    
    if not (net and output_layer_names and appsrc_global):
        # YOLO not loaded or appsrc not ready, skip
        sample = sink.emit("pull-sample")
        return Gst.FlowReturn.OK
    
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps()
    
    # Extract frame data
    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.OK
    
    frame_data = mapinfo.data
    # Determine width, height from caps
    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")
    
    # Convert raw bytes to numpy array
    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
    
    # Run YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layer_names)
    
    boxes, confidences, classIDs = [], [], []
    h, w = frame.shape[:2]

    # Thresholds
    conf_thresh = 0.5
    nms_thresh = 0.3

    for output in detections:
        for det in output:
            scores = det[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf_thresh:
                box = det[0:4] * np.array([w, h, w, h])
                (centerX, centerY, bwidth, bheight) = box.astype("int")
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # Non-max suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            text = f"Person" if (classIDs[i] == 0) else f"Class {classIDs[i]}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Push the modified frame to appsrc
    # Convert frame back to GStreamer buffer
    new_buf = Gst.Buffer.new_allocate(None, frame.size, None)
    new_buf.fill(0, frame.tobytes())
    
    # Set timestamp (optional but recommended)
    new_buf.pts = buf.pts
    new_buf.dts = buf.dts
    
    # Unmap original buffer
    buf.unmap(mapinfo)
    
    # Push new buffer to appsrc
    retval = appsrc_global.emit("push-buffer", new_buf)
    return Gst.FlowReturn.OK

###############################################################################
# 5. START / STOP RTSP SERVER
###############################################################################

def start_rtsp_server():
    """
    Start the RTSP server with the custom pipeline.
    """
    global rtsp_server, main_loop
    with pipeline_lock:
        if rtsp_server is not None:
            print("[INFO] RTSP server already running.")
            return
        
        rtsp_server = GstRtspServer.RTSPServer()
        factory = CustomRtspMediaFactory(video_file=VIDEO_FILE_PATH)
        factory.set_shared(True)
        
        mount_points = rtsp_server.get_mount_points()
        mount_points.add_factory("/test", factory)
        
        rtsp_server.attach(None)
        
        main_loop = GLib.MainLoop()
        
        # Run GLib loop in a separate thread
        def loop_func():
            try:
                main_loop.run()
            except:
                pass
        
        t = threading.Thread(target=loop_func, daemon=True)
        t.start()
        print("[INFO] RTSP server started. Stream at rtsp://<IP>:8554/test")

def stop_rtsp_server():
    """
    Stop the RTSP server if running.
    """
    global rtsp_server, main_loop
    with pipeline_lock:
        if rtsp_server is None:
            print("[INFO] No RTSP server to stop.")
            return
        
        if main_loop is not None:
            main_loop.quit()
        
        rtsp_server = None
        main_loop = None
        print("[INFO] RTSP server stopped.")

###############################################################################
# 6. FLASK REST API
###############################################################################
@app.route("/start", methods=["POST"])
def api_start():
    """
    Start the RTSP stream
    """
    start_rtsp_server()
    return jsonify({"status": "streaming_started"}), 200

@app.route("/stop", methods=["POST"])
def api_stop():
    """
    Stop the RTSP stream
    """
    stop_rtsp_server()
    return jsonify({"status": "streaming_stopped"}), 200

@app.route("/status", methods=["GET"])
def api_status():
    """
    Return whether the server is running
    """
    status = "running" if rtsp_server else "stopped"
    return jsonify({"status": status}), 200

@app.route("/enable_detection", methods=["POST"])
def api_enable_detection():
    """
    Enable YOLO detection on each frame (bonus feature).
    """
    global object_detection_enabled
    if not OPENCV_AVAILABLE:
        return jsonify({"error": "OpenCV (cv2) not installed"}), 400
    
    object_detection_enabled = True
    load_yolo_model()
    return jsonify({"status": "object_detection_enabled"}), 200

@app.route("/disable_detection", methods=["POST"])
def api_disable_detection():
    """
    Disable YOLO detection
    """
    global object_detection_enabled
    object_detection_enabled = False
    return jsonify({"status": "object_detection_disabled"}), 200

###############################################################################
# 7. OPTIONAL: MQTT SETUP
###############################################################################
def on_mqtt_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC_CONTROL)

def on_mqtt_message(client, userdata, msg):
    # Handle inbound MQTT messages for controlling the stream
    message = msg.payload.decode("utf-8")
    print(f"[MQTT] Received on {msg.topic}: {message}")
    
    if message == "start":
        start_rtsp_server()
        mqtt_client.publish(MQTT_TOPIC_STATUS, "Stream started")
    elif message == "stop":
        stop_rtsp_server()
        mqtt_client.publish(MQTT_TOPIC_STATUS, "Stream stopped")
    else:
        print("[MQTT] Unknown command.")

def start_mqtt():
    global mqtt_client
    if not MQTT_AVAILABLE:
        print("[INFO] paho-mqtt not installed, skipping MQTT.")
        return
    
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    # Start MQTT network loop in a separate thread
    mqtt_thread = threading.Thread(target=mqtt_client.loop_forever, daemon=True)
    mqtt_thread.start()
    print("[INFO] MQTT client started.")

###############################################################################
# 8. MAIN ENTRY POINT
###############################################################################
if __name__ == "__main__":
    # If you want MQTT enabled by default, uncomment below:
    # start_mqtt()
    
    # Uncomment if you will use YOLO
    # if OPENCV_AVAILABLE:
    #     load_yolo_model()
    
    # Start Flask
    app.run(host="0.0.0.0", port=5000, debug=False)
