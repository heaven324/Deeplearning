import numpy as np
import two_layer_net as tln

x = np.random.randn(10, 2)
model = tln.TwoLayerNet(2,4,3)
s = model.predict(x)
print(s)