import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression

def get_close_values(name):
    cwd = os.getcwd() + '/'
    path = cwd + name
    file = pd.read_csv(path)
    close = file['Close']
    return close

def moving_average_convergence_divergence(close, upper, lower, mid):
    ema_upper_bound = pd.Series.ewm(close, span=upper).mean()
    ema_lower_bound = pd.Series.ewm(close, span=lower).mean()
    diff = ema_lower_bound - ema_upper_bound
    macd = pd.Series.ewm(diff, span=mid).mean()
    return macd

def grid_plot(price, indicator):
    grid = plt.GridSpec(3, 1, wspace=0.4, hspace=0.3)
    plt.subplot(grid[0:2,:])
    plt.plot(price)
    plt.subplot(grid[2:, :])
    # make macd zero line
    zero_line = []
    for i in range(len(indicator)):
        zero_line.append(0)
    plt.plot(indicator)
    plt.plot(zero_line, color='gray', alpha=0.5)
    plt.show()

def separate_price_by_macd(close, macd):
    # per var
    separated_price_by_macd = []
    peaks = []
    peaks_pos = []

    # temp var
    i = 1 # because MACD starts at index 1
    temp_list = [] # each seperated price by macd stored in here
    # because we start at index 1:
    temp_list.append(close[0])

    while i<len(macd):
        if (macd[i] > 0 and macd[i-1] >= 0) or (macd[i] < 0 and macd[i-1] <= 0):
            # continouse MACD
            if i==len(macd)-1:
                # last series
                temp_list.append(close[i])
                if macd[i-1] > 0:
                    peaks.append(max(temp_list))
                    peaks_pos.append(i - (len(temp_list) - temp_list.index(max(temp_list))))
                else:
                    peaks.append(min(temp_list))
                    peaks_pos.append(i - (len(temp_list) - temp_list.index(min(temp_list))))
                separated_price_by_macd.append(temp_list)
                i = i + 1
            else:
                temp_list.append(close[i])	
                i = i + 1
        elif (macd[i] > 0 and macd[i-1] < 0) or (macd[i] < 0 and macd[i-1] > 0):
            # MACD phase changed
            if macd[i-1] > 0:
                peaks.append(max(temp_list))
                # print(max(temp_list), close[i - (len(temp_list) - temp_list.index(max(temp_list)))])
                peaks_pos.append(i - (len(temp_list) - temp_list.index(max(temp_list))))
            else:
                peaks.append(min(temp_list))
                # print(min(temp_list), close[i - (len(temp_list) - temp_list.index(min(temp_list)))])
                peaks_pos.append(i - (len(temp_list) - temp_list.index(min(temp_list))))
            separated_price_by_macd.append(temp_list)
            temp_list = []
            temp_list.append(close[i])
            i = i + 1
    return separated_price_by_macd, peaks, peaks_pos

def plot_peaks(price, peaks, peaks_pos, indicator):
    grid = plt.GridSpec(3, 1, wspace=0.4, hspace=0.3)
    plt.subplot(grid[0:2,:])
    plt.plot(price)
    plt.scatter(peaks_pos, peaks, color='red')
    plt.title('Scatter plot peaks')
    plt.subplot(grid[2:, :])
    # make macd zero line
    zero_line = []
    for i in range(len(indicator)):
        zero_line.append(0)
    plt.plot(indicator)
    plt.plot(zero_line, color='gray', alpha=0.5)
    plt.show()

def separated_price_by_38_percant(close, peaks, peaks_pos):
    # per var
    deletation_num = 0
    new_close = close
    new_peaks = peaks
    new_peaks_pos = peaks_pos

    #temp var
    temp_len = 0

    while temp_len < len(close):
        if abs(len(close[temp_len-1])-len(close[temp_len-2])) == 0 or abs(peaks[temp_len-1]-peaks[temp_len-2]) == 0:
            # 100% time or price correction(devision by zero)
            temp_len = temp_len + 1
        elif (abs(len(close[temp_len])-len(close[temp_len-1]))/abs(len(close[temp_len-1])-len(close[temp_len-2]))) > 0.38 or (abs(peaks[temp_len]-peaks[temp_len-1])/abs(peaks[temp_len-1]-peaks[temp_len-2])) > 0.38:
            # above 38% time or price correction
            temp_len = temp_len + 1
        else:
            # below 38% time or price correction
            if temp_len+2<=len(close):
                temp_list = []
                new_list = []
                peak = 0
                peak_pos = 0
                new_peaks_list = []
                new_peaks_pos_list = []

                if macd[temp_len]>0:
                    # close[temp_len] is positive peak
                    if close[temp_len]>close[temp_len+2]:
                        peak = peaks[temp_len]
                        peak_pos = peaks_pos[temp_len]
                    else:
                        peak = peaks[temp_len+2]
                        peak_pos = peaks_pos[temp_len+2]
                else:
                    # close[temp_len] is negative peak 
                    if close[temp_len]<close[temp_len+2]:
                        peak = peaks[temp_len]
                        peak_pos = peaks_pos[temp_len]
                    else:
                        peak = peaks[temp_len+2]
                        peak_pos = peaks_pos[temp_len+2]

                for i in range(temp_len, temp_len+3):
                    temp_list.extend(close[i])

                i=0
                while i < len(new_close):
                    if i==temp_len-(deletation_num*2):
                        new_list.append(temp_list)
                        new_peaks_list.append(peak)
                        new_peaks_pos_list.append(peak_pos)
                        i = i + 3
                    else:
                        new_list.append(new_close[i])
                        new_peaks_list.append(new_peaks[i])
                        new_peaks_pos_list.append(new_peaks_pos[i])
                        i = i + 1

                deletation_num = deletation_num + 1
                new_close = new_list
                new_peaks = new_peaks_list
                new_peaks_pos = new_peaks_pos_list
                temp_len = temp_len + 3
            else:
                temp_len = temp_len + 1
    return new_close, new_peaks, new_peaks_pos

def peaks_to_chart(close, peaks):
    # per var
    final_close = []

    #temp var
    temp_peaks = []

    line_length = close[0].index(peaks[0]) + 1 # first part
    temp_peaks.append(close[0][0])
    temp_peaks.append(peaks[0])
    x = np.linspace(temp_peaks[0], temp_peaks[1], line_length).tolist()
    for t in range(len(x)):
        final_close.append(x[t])
    line_length = len(close[0]) - close[0].index(peaks[0]) # second part
    temp_peaks.pop(0)
    temp_peaks.append(peaks[1])

    for i in range(1, len(close)):
        line_length = line_length + close[i].index(peaks[i]) + 1 # first part
        x = np.linspace(temp_peaks[0], temp_peaks[1], line_length).tolist()
        x.pop(0)
        temp_peaks.pop(0)
        for t in range(len(x)):
            final_close.append(x[t])
        line_length = len(close[i]) - close[i].index(peaks[i]) # second part
        if i < len(close) - 1:
            temp_peaks.append(peaks[i+1])
        else:
            temp_peaks.append(close[-1][-1])
    x = np.linspace(temp_peaks[0], temp_peaks[1], line_length).tolist()
    x.pop(0)
    for t in range(len(x)):
        final_close.append(x[t])
    return final_close

def plot_majors(close, majors):
    plt.plot(close)
    plt.plot(majors)
    plt.show()

def find_trend_lines(peaks, peaks_pos, close):
    for i in range(len(peaks)):
        if i <= len(peaks) - 5:
            m = (peaks[i+2] - peaks[i]) / (peaks_pos[i+2] - peaks_pos[i])
            y = (m * (peaks_pos[i+4] - peaks_pos[i])) + peaks[i]
            percentage = 0.03
            if y - (percentage * y) < peaks[i+4] < y + (percentage * y):
                # It's a line so check if it's continouse
                trend_peaks = []
                trend_peaks.append(peaks[i])
                trend_peaks.append(peaks[i+2])
                trend_peaks.append(peaks[i+4])
                trend_peaks = np.array(trend_peaks).reshape(-1,1)
                trend_peaks_pos = []
                trend_peaks_pos.append(peaks_pos[i])
                trend_peaks_pos.append(peaks_pos[i+2])
                trend_peaks_pos.append(peaks_pos[i+4])
                trend_peaks_pos = np.array(trend_peaks_pos).reshape(-1,1)
                linear_regression(trend_peaks_pos, trend_peaks, close)

def linear_regression(day, price, close):
    model = LinearRegression(fit_intercept=True)
    model.fit(day, price)
    day_fit = np.linspace(day[0], day[-1]+(day[-1]-day[0]),len(price) + 1).reshape(-1,1)
    price_fit = model.predict(day_fit)
    plt.plot(close)
    plt.plot(day_fit, price_fit)
    plt.show()


close = get_close_values('file.csv')
macd = moving_average_convergence_divergence(close, 13, 6, 1) # 26, 12, 9 => mid-term 13, 6, 1 => short-term 52, 24, 18 => long-term
# grid_plot(close, macd)
separated_price_by_macd, peaks, peaks_pos = separate_price_by_macd(close, macd)
# print(sum(len(x) for x in separated_price_by_macd), len(close))
# plot_peaks(close, peaks, peaks_pos, macd)
semi_final_close, final_peaks, final_peaks_pos = separated_price_by_38_percant(separated_price_by_macd, peaks, peaks_pos)
# print(f'Length of peaks before exerting 38%: {len(peaks)}\n', f'Length of peaks after exerting 38%: {len(final_peaks)}')
# plot_peaks(close, final_peaks, final_peaks_pos, macd)
# print(sum(len(x) for x in semi_final_close), len(close))
final_close = peaks_to_chart(semi_final_close, final_peaks)
# plot_majors(close, final_close)
find_trend_lines(final_peaks, final_peaks_pos, close)
