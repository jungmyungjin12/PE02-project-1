import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml")
root = tree.getroot()

for i in root.iter('Current'):
    a = i.text
for i in root.iter('Voltage'):
    b = i.text

I = (list(map(float,a.split(','))))
V = (list(map(float,b.split(','))))
print(V)
print(I)
I1 = np.abs(I)

plt.plot(V, I1, 'bo--')
plt.title('IV analysis', fontdict = {'weight': 'bold', 'size':15})
plt.xlabel('Voltage[V]', labelpad=8 , fontdict={'weight': 'bold', 'size':12})
plt.ylabel('Current[I]', labelpad=8 , fontdict={'weight': 'bold', 'size':12})
plt.yscale('logit')
plt.grid(True)
plt.show()