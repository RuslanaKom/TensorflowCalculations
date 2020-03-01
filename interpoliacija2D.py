from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np

# tf.logging.set_verbosity(tf.logging.ERROR)

data = np.array([
                 [1,1], [2,1], [3,1], [4,1],
                 [1,6], [2,6], [3,6], [4,6],
                 [1,9], [2,9], [3,9], [4,9],
                 [1,12], [2,12], [3,12], [4,12],
                 [1,15], [2,15], [3,15], [4,15],
                 [1,18], [2,18], [3,18], [4,18],
                 [1,24], [2,24], [3,24], [4,24],
                 [1,30], [2,30], [3,30], [4,30],
                 [1,40], [2,40], [3,40], [4,40],
                 [1,50], [2,50], [3,50], [4,50]
                 ], dtype=float)

koef = np.array([1.0, 1.0, 1.0, 1.0,
                 0.791, 0.5, 0.515, 0.576,
                 0.602, 0.376, 0.412, 0.452,
                 0.499, 0.312, 0.353, 0.394,
                 0.440, 0.269, 0.316, 0.368,
                 0.396, 0.250, 0.294, 0.348,
                 0.340, 0.228, 0.273, 0.316,
                 0.310, 0.214, 0.256, 0.288,
                 0.283, 0.197, 0.236, 0.264,
                 0.264, 0.185, 0.221, 0.240
                 ], dtype=float)

for i, c in enumerate(data):
    print("{} params = {} koef".format(c, koef[i]))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, activation="tanh", input_shape=[2], kernel_initializer="uniform"))
model.add(tf.keras.layers.Dense(1, activation="linear", kernel_initializer="uniform"))



model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
history = model.fit(data, koef, epochs=1000, verbose=False)
print("Training finished")
plt2.xlabel('Epoch')
plt2.ylabel('Loss')
plt2.plot(history.history['loss'])
plt2.show()

# create an index for each tick position
xi = list(range(len(data)))

plt.plot(xi, koef, marker='o', linestyle='--', color='r', label='data')
plt.plot(xi, model.predict(data), marker='o', linestyle='--', color='g', label='prediction')
plt.xlabel('quantity')
plt.ylabel('koeficient')
plt.xticks(xi, data)
plt.title('compare')
plt.legend()
plt.show()

while(1):
    bulding_type = input("Enter building type:")
    quantity = input("Enter quantity:")
    print(model.predict(np.array([[int(bulding_type), int(quantity)]])))
