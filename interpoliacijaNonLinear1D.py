from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

data = np.array([1, 6, 9, 12, 15, 18, 24, 30, 40, 50, 60, 70, 80, 90, 100, 120], dtype=float)

koef = np.array(
   [1, 0.791, 0.602, 0.499, 0.440, 0.396,  0.340, 0.310, 0.283, 0.264, 0.249, 0.238, 0.228, 0.220, 0.213, 0.201],
  dtype=float)


for i, c in enumerate(data):
    print("{} params = {} koef".format(c, koef[i]))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(20, activation="tanh", input_shape=[1], kernel_initializer="uniform"))
model.add(tf.keras.layers.Dense(1, activation="linear", kernel_initializer="uniform"))

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))

history = model.fit(data, koef, epochs=500, verbose=False)

print("Training finished")
print(history.history['loss'])
# print("Weight of the layer l0: {}".format(l0.get_weights()))

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
  value_to_predict = input("Enter value:")
  print(model.predict(np.array([int(value_to_predict)])))
