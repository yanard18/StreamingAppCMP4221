import cv2

def main():
    # Change the IP/port if necessary
    rtsp_url = "rtsp://127.0.0.1:8554/test"
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("[ERROR] Could not open RTSP stream.")
        return
    
    print("[INFO] Receiving RTSP stream. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended or cannot retrieve frame.")
            break
        
        cv2.imshow("RTSP Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
