


## 1. Features


- **Video Compression**
Uses x264enc for compressing the video stream (bitrate=500, speed-preset=superfast, tune=zerolatency) to optimize bandwidth.

- **Person Detection** (Optional)
A YOLO detection pipeline (commented out by default) can be enabled to detect class ID 0 (person) or any other specified class. Bounding boxes are drawn on the stream if enabled.

- **MQTT** (Optional)
You can optionally control the server (start/stop streaming) using MQTT messages.


## 2. Prerequisites
1. **Python 3**
	- Make sure Python 3 is installed on your system.

2. **GStreamer**
	- Install GStreamer and its plugins. The instructions below focus on Ubuntu/Debian. For Windows/macOS, see the System Dependencies section.


## 3. Installatuion Guide


### 3.1 System Dependencies (Linux)

For Ubuntu/Debian-based systems, run:

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

sudo apt-get install libgstrtspserver-1.0-dev gstreamer1.0-rtsp
sudo apt-get install build-essential meson ninja-build pkg-config libffi-dev \
    libgirepository1.0-dev libcairo2-dev
```

### 3.2 System Dependencies (Windows/Mac)

**Windows**
	- Download and install GStreamer from gstreamer.freedesktop.org.
    - Install Python 3 from python.org.
    - (Optional) Use a package manager like Chocolatey to simplify the process.

**macOS**
	- Install GStreamer via Homebrew:

### 3.3 Create a Virtual Environment (Optional)

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

### 3.4 Install Python Dependencies

First, upgrade `pip`, `setuptools`, and `wheel` to the latest versions:

```bash
pip3 install --upgrade pip setuptools wheel
```

Next, install the required Python packages using the provided `requirements.txt`:


```bash
pip3 install -r requirements.txt
```

## 4. Optional Setup for YOLO Detection

If you want to enable YOLO object detection in the stream, download the YOLOv3 configuration and weights files (yolov3.cfg and yolov3.weights) and place them in the same directory as the project files. You can download them from the official YOLO website.

## 5. Optional Setuo for MQTT

If you want to use MQTT to control the stream, install and run an MQTT broker such as Mosquitto:

```bash
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

# 6. Quick Start

This part aims to show how you can start using the rtsp server with most possible less reading.

1. *Run the Server:*

```bash
python3 server.py
```

2. **Start the RTSP stream** (via REST)

```bash
curl -X POST http://<server-ip>:5000/start
```

3. **Run the client for sending GET request:**

```bash
python3 client.py
```

## 7 Usage Instructions

### 7.1 REST Endpoints

*Endpoints:*
- `POST /start` : Start the RTSP stream.
- `POST /stop` : Stop the RTSP stream.
- `GET /status` : Check if streaming.
- `POST /enable_detection` : Enable YOLO detection (if OpenCV + YOLO are installed).
- `POST /disable_detection` : Disable YOLO detection.

Example usage:

```bash
curl -X POST http://localhost:5000/start
```

```bash
curl http://localhost:5000/status
```

### 7.2 MQTT

If MQTT is configured, the server listens on (for example) stream/control topic: (*uncommnet the source code*)

```bash
mosquitto_pub -t stream/control -m "start"
mosquitto_pub -t stream/control -m "stop"
```

---

>	Note: The default compression settings use x264enc with bitrate=500, speed-preset=superfast, and tune=zerolatency to optimize bandwidth. Adjust as needed for your specific use case.


