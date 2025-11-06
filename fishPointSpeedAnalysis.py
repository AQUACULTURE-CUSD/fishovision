import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ERROR: relative path not working
df = pd.read_csv('/Users/lieselwong/Documents/aquaculture/fishovision/averageFishPointSpeed.csv', names = ["Frames", "Average Fish Point Speed"])
df.head()

plt.figure(figsize=(12, 6))
plt.plot(df['Frames'], df['Average Fish Point Speed'])
plt.xlabel('Frames')
plt.ylabel('Average Fish Point Speed')
plt.title('Average Fish Point Speed Over Time')
plt.grid(True)
plt.show()