#!/bin/python3

import math
import os
import random
import re
import sys
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm


#
# Complete the 'predictTemperature' function below.
#
# The function is expected to return a FLOAT_ARRAY.
# The function accepts following parameters:
#  1. STRING startDate
#  2. STRING endDate
#  3. FLOAT_ARRAY temperature
#  4. INTEGER n
#

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def training(df, duration, n):
    res_df = pd.DataFrame()

    print(res_df)

    for idx in range(24):
        model = ARIMA(df.iloc[[idx]].T.values, order=(2, 1, 1))
        model_fit = model.fit(disp=0)
        # model = pm.auto_arima(df.iloc[[idx]].T.values, start_p=1, start_q=1,
        #                       test='adf',
        #                       max_p=3, max_q=3,
        #                       d=None,
        #                       m=12,
        #                       seasonal=True,
        #                       start_P=0,
        #                       D=0,
        #                       trace=True,
        #                       error_action='ignore',
        #                       suppress_warnings=True,
        #                       stepwise=True)
        #     print(model_fit.summary())
        #     residuals = pd.DataFrame(model_fit.resid)
        #     fig, ax = plt.subplots(1,2)
        #     residuals.plot(title="Residuals", ax=ax[0])
        #     residuals.plot(kind='kde', title='Density', ax=ax[1])
        #     plt.show()

        #     model_fit.plot_predict(dynamic=False)
        #     plt.show()

        prediction = model_fit.predict(duration, duration + n - 1)
        res = inverse_difference(df.iloc[[idx]].T.values, prediction)

        res_df[idx] = res

    return res_df


def predictTemperature(startDate, endDate, temperature, n):
    # if only one day of data is present, just return the data as it will not provide moving avg.
    if len(temperature) == 24:
        return temperature

    start = datetime.strptime(startDate, '%Y-%m-%d')
    end = datetime.strptime(endDate, '%Y-%m-%d') + timedelta(hours=23)
    duration = len(temperature) // 24
    cur = start

    past_temperature = defaultdict()
    i = 0

    while cur <= end:
        date = '-'.join([str(cur.year), str(cur.month), str(cur.day)])
        hr = str(cur.hour)
        if date not in past_temperature:
            past_temperature[date] = defaultdict()
        past_temperature[date][int(hr)] = temperature[i]

        cur += timedelta(hours=1)
        i += 1

    past_temp_df = pd.DataFrame.from_dict(past_temperature)
    #     print(past_temp_df)

    res = training(past_temp_df, duration=duration, n=n)
    print(past_temp_df)
    print(res)

    ans = []
    for idx, row in res.iterrows():
        for i in row.values.tolist():
            ans.append(i)

    return ans


if __name__ == '__main__':
    #     fptr = open(os.environ['OUTPUT_PATH'], 'w')
    inp = open('input001.txt', 'r')

    startDate = inp.readline().strip()

    endDate = inp.readline().strip()

    temperature_count = int(inp.readline().strip())

    temperature = []

    for _ in range(temperature_count):
        temperature_item = float(inp.readline().strip())
        temperature.append(temperature_item)

    n = int(inp.readline().strip())

    result = predictTemperature(startDate, endDate, temperature, n)

#     fptr.write('\n'.join(map(str, result)))
#     fptr.write('\n')

#     fptr.close()


