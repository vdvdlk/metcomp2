import matplotlib.pyplot as plt
import scipy.misc

face = scipy.misc.face()
print(face.shape)
print(face.max())
print(face.dtype)


plt.gray()
plt.imshow(face)
plt.show()
