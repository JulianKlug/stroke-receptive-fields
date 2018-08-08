import matplotlib.pyplot as plt

data = [
    {
      'rf':0,
      '5CV_time': 1278.5785219334066,
      '5CV_size': 0.483,
    },
    {
      'rf':1,
      '5CV_time': 2943.732852779329,
      '5CV_size': 13,
    },
    {
      'rf':2,
      '5CV_time': 12407.027481898665,
      '5CV_size': 66,
    },
    {
      'rf':3,
      '5CV_time': 28565.381308908574,
      '5CV_size': 167,
    },
    {
      'rf':4,
      '5CV_time': 60371.71249007899,
      '5CV_size': 365,
    },
    {
      'rf':5,
      '5CV_time': 102878.19795801304,
      '5CV_size': 673,
    }
]

rf = list(map(lambda x: x['rf'], data))
CV_time = list(map(lambda x: x['5CV_time'], data))
CV_size = list(map(lambda x: x['5CV_size'], data))

fig, ax1 = plt.subplots()
ax1.plot(rf, CV_time, 'C0', label=r'Cross-Validation time')
ax1.set_ylabel('time [s]', color='C0')

ax2 = ax1.twinx()
ax2.plot(rf, CV_size, 'r', label=r'Cross-Validation data size')
ax2.set_ylabel('data size [GB]', color='r')

bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="black", lw=0.3)
t = ax1.text(4.2, 100878, "CV time: 28.6h", ha="center", va="center", rotation=0,
            size=10,
            bbox=bbox_props)

ax1.set_xlabel('Receptive field size (as voxels from center)')
plt.title('Metaparameters')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="lower right")

plt.ion()
plt.draw()
plt.show()
