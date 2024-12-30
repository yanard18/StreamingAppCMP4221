## Bonus Points

Video Compression:
We already used x264enc in the pipeline to compress the video stream (bitrate=500, speed-preset=superfast, tune=zerolatency) to optimize bandwidth.

MQTT:
Empty yet

Person Detection (Object Detection):
optional YOLO pipeline inside the code commented. The bounding box is drawn if the YOLO class is “person” (class ID 0) or any other class.

Demonstration Video:
Record a short screencast showing the server, the REST calls, the client receiving the video, and (optionally) bounding boxes for object detection.

## Usage Instructions

1. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
pip3 install flask
# Optional:
pip3 install paho-mqtt opencv-python
```

If you want YOLO detection, place yolov3.cfg and yolov3.weights in the same directory.
If you want MQTT, also install and run an MQTT broker like mosquitto:

```bash
sudo apt-get install mosquitto
sudo service mosquitto start
```

Run the server:

```bash
python3 server.py
```

*Endpoints:*
- `POST /start` : Start the RTSP stream.
- `POST /stop` : Stop the RTSP stream.
- `GET /status` : Check if streaming.
- `POST /enable_detection` : Enable YOLO detection (if OpenCV + YOLO are installed).
- `POST /disable_detection` : Disable YOLO detection.

```bash
curl -X POST http://localhost:5000/start
```

```bash
curl http://localhost:5000/status
```

Optional MQTT:

```bash
mosquitto_pub -t stream/control -m "start"
mosquitto_pub -t stream/control -m "stop"
```

## Use VENV when working

If you don't want tons of random python library on your computer use VENV

1. Navigate to your project directory:

```bash
cd /path/to/your/project
```

2. Create a Virtual Environment

```bash
python3 -m venv venv
```

3. Activate the virtual environment:

	- On Linux/Mac:
	```bash
	source venv/bin/activate
	```

	- On Windows CMD:
	```bash
	venv\Scripts\activate
	```



