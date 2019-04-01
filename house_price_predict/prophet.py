import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
from utils import *

city_price_dict = load_data()
city_price_dict = wash_dict(city_price_dict, True)
x_vals = list()
y_vals = list()
for i, x in enumerate(city_price_dict["北京"]):
    if i == 0:
        continue
    x_vals.append(x[0])
    y_vals.append(x[1])
x_vals = recover(np.array(x_vals))  # 时间
y_vals = np.array(y_vals)

df = list()
df = df.append(x_vals)
df = df.append(y_vals)
df = pd.DataFrame(df, columns=['ds', 'y'])
df.head()
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
m.plot(forecast)

x1 = forecast['ds']
y1 = forecast['yhat']


plt.plot(x1,y1)
plt.show()