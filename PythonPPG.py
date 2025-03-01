import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# signal smoothing using savgol filter 
def smooth_signal(ppg_signal, window_length=5, polyorder=2):
    return savgol_filter(ppg_signal, window_length, polyorder)

# Preprocessing
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
    threshold = 0.2 * np.max(integrated_signal)
    peaks = np.where(integrated_signal > threshold)[0]
    verified_peaks = []
    refractory_period = 0.5  

    for peak in peaks:
        if not verified_peaks or (peak - verified_peaks[-1]) > refractory_period:
            verified_peaks.append(peak)

    heart_rate = 60 / np.mean(np.diff(verified_peaks))  
    return heart_rate

# Capture video from a file
cap = cv2.VideoCapture('video2.mp4')

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

ppg_data = smooth_signal(ppg_data)
# Perform PPG signal processing
preprocessed_signal = preprocess_ppg(ppg_data, lowcut=3, highcut=10, order=5)
heart_rate = detect_peaks(preprocessed_signal)

#heart rate
print("Heart Rate (BPM):", heart_rate)

# Plot the PPG signal and detected peaks (optional)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ppg_data, label='Raw PPG Signal', color='b')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Raw PPG Signal')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(preprocessed_signal, label='Preprocessed Signal', color='r')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Preprocessed PPG Signal')
plt.legend()


plt.tight_layout()
plt.show()
