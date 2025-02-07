# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import yfinance as yf


# Data Acquisition and Preparation(دریافت و آماده‌سازی داده‌ها)
df = yf.download("BTC-USD")
df = df[:900]
y = df["Close"].values.reshape(-1,1)
X = np.arange(len(y)).reshape(-1,1)

# Hyperparameter Definition and Initialization( تعریف و مقداردهی اولیه ابرپارامترها)
learning_rate = 0.000001
epochs = 1500

X_bias = np.c_[np.ones_like(X),X]#(اضافه کردن یک جمله بایاس به x)
samples, features = X_bias.shape
theta = np.zeros((features,1)) #(مقداردهی اولیه پارامترهای مدل (وزن‌ها))
cost_function = [] #(لیستی برای ذخیره مقدار تابع هزینه در هر تکرار)



# Gradient Descent Function (within animate)
def animate(frame):
    global X_bias,theta,learning_rate,samples,cost_function
    h = np.dot(X_bias, theta)#فرضیه(پیشبینی)
    error = h - y# خطا (اختلاف بین پیش‌بینی و مقدار واقعی)
    gradient = np.dot(X_bias.T, error) / samples #محاسبه گرادیان
    print(gradient)
    print(f"y = {theta[1][0]:6}. x + {theta[0][0]:6}")#نمایش معادله رگرسیون خطی
    theta -= learning_rate * gradient #به روز رسانی پارامتر ها
    mean_squared_error = np.mean(np.square(error)) #محاسبه mse
    cost_function.append(mean_squared_error)
    line1.set_ydata(h)
    line2.set_xdata(np.arange(len(cost_function)))
    line2.set_ydata(cost_function)
    ax2.set_ylim(0, max(cost_function) + max(cost_function) * 0.1)
    return line1, line2

 # Plotting and Animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(X, y)

line1, = ax1.plot(X, y)
line2, = ax2.plot([], [], lw=2, color="red")

ax1.set_xlim(-10, len(X))
ax1.set_ylim(-100, y.max() + y.max() * 0.1)

ax2.set_xlim(-10, epochs)

ani = FuncAnimation(fig, animate, frames=epochs, interval=1)
plt.show()