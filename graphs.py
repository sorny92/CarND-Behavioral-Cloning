import matplotlib.pyplot as plt
import csv
import numpy as np

samples = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)

samples = np.asarray(samples)
print(samples[1000])

plt.hist(samples[:,3].astype(np.float), bins=50)

#plt.hist(samples)
plt.show()

