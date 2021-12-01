import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from scipy.signal import find_peaks
import numpy as np
import time


# data filter
def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    normal_cutoff = float(cutoff_freq) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    y = signal.filtfilt(b, a, data)
    return y


# find peaks and valleys, unoptimised!
def real_time_validation(x, data, thresh=0.9):
    peak_idx, _ = find_peaks(data, height=thresh)
    valley_idx, _ = find_peaks(-data)

    if len(peak_idx) > 0:
        ax.plot(x[peak_idx[0]], data[peak_idx[0]], 'r.')
    else:
        peak_idx = 'X'

    if len(valley_idx) > 0:
        ax.plot(x[valley_idx[0]], data[valley_idx[0]], 'b.')
    else:
        valley_idx = 'X'

    return peak_idx[0], valley_idx[0]


df = pd.read_csv('../data/testwalk.csv')

# create 'canvas'
matplotlib.use('TKAgg', force=True)
fig = plt.figure()
ax = fig.add_subplot(111)

times = []
wz = []
li, = ax.plot(times, wz, '.', c='black')

# draw and show it
fig.canvas.draw()
plt.show(block=False)

n_p = 0
k = 0

last_phase = 'unknown'
current_phase = 'unknown'

for n in range(10, 10000, 10):

    # read 10 rows every iteration: simulate real time
    dataFrame = df[n_p:n]
    times = np.append(times, dataFrame['time'], axis=0)
    wz = np.append(wz, dataFrame['wz (rad/s)'], axis=0)

    # initial optimisation
    if n > 50 and n % 2 == 0:
        wz = np.append(wz[:n-50],
                       butter_lowpass_filter(wz[n-50:], 1, 40), axis=0)

        peak_idx, valley_idx = real_time_validation(times[n-50:], wz[n-50:])

        # prevent finding multiple peaks in 50 'ms' window
        if n - k > 50:
            if(peak_idx != 'X' and
               (current_phase == 'unknown' or current_phase == 'step')):
                current_phase = 'swing'

            elif(valley_idx != 'X' and current_phase == 'swing'):
                current_phase = 'step'

            elif(valley_idx != 'X' and current_phase == 'step'):
                current_phase = 'swing'

        if peak_idx != 'X' or valley_idx != 'X':
            k = n

    print(current_phase)
    n_p = n

    # set the new data
    li.set_xdata(times)
    li.set_ydata(wz)
    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.canvas.draw()

    # 10 ms
    time.sleep(0.01)

plt.show()
