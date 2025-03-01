from flask import Flask, request, jsonify
from flask_cors import CORS,  cross_origin
import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

@app.route("/test", methods=["GET"])
def test():
    return "Hello World!"

 
def preprocess_ppg(ppg_signal, lowcut, highcut, order):
    nyquist = 0.5 * 100
    lowcut /= nyquist
    highcut /= nyquist
    b, a = signal.butter(order, [lowcut, highcut], btype='band')
    return signal.lfilter(b, a, ppg_signal)

# Peak detection and heart rate estimation using Pan-Tompkins algorithm
def detect_peaks(ppg_signal):
    diff_signal = np.diff(ppg_signal)
    squared_signal = diff_signal ** 2
    integrated_signal = np.convolve(squared_signal, np.ones(10), 'same')
    threshold = 0.5 * np.max(integrated_signal)
    peaks = np.where(integrated_signal > threshold)[0]
    verified_peaks = []
    refractory_period = 0.3  

    for peak in peaks:
        if not verified_peaks or (peak - verified_peaks[-1]) > refractory_period:
            verified_peaks.append(peak)

    heart_rate = 60 / np.mean(np.diff(verified_peaks))  
    return verified_peaks, heart_rate

@app.route("/ppg", methods=["POST"])
@cross_origin()
def process_video():
    try:
        # Check if the request contains a file named 'file'
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        # Check if the file has an allowed extension (e.g., .mp4)
        allowed_extensions = {"mp4"}
        if "." not in file.filename or file.filename.split(".")[-1].lower() not in allowed_extensions:
            return jsonify({"error": "Invalid file extension"}), 400

        # Save the uploaded video temporarily
        video_path = "t_video.mp4"  
        file.save(video_path)

        # Process the video
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()

            if ret:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
                # Define range for blue color in HSV
                lower_blue = np.array([0,50,50])
                upper_blue = np.array([180,255,255])
        
                # Threshold the HSV image to get only blue colors
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
                # Bitwise-AND mask and original image
                res = cv2.bitwise_and(frame, frame, mask=mask)
        
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        
                frames.append(gray)
            else:
                break
            
        video_array = np.array(frames)

        # Process video_array as PPG data
        ppg_data = np.mean(video_array, axis=(1, 2))  

        # Perform PPG signal processing
        preprocessed_signal = preprocess_ppg(ppg_data, lowcut=0.5, highcut=10, order=5)
        peaks, heart_rate   = detect_peaks(preprocessed_signal)

        # Return the result as JSON response
        result = {"heart_rate": heart_rate}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": app.logger.error(str(e))}), 500

if __name__ == "__main__":
    app.run(debug=True)
