import pandas
from numpy import arange, random
from cellshift import CS

random.seed(25)
X = arange(1, 101)
Y_base = (1/4) * X + 20
noise = random.uniform(-5, 5, size=len(X))
d = CS( pandas.DataFrame({'X': X, 'Y': Y_base + noise}) )
d.data.limit(6).show()
d.add_salt_pepper_noise_column("Y", "noise_Y", sample_pct=20)
d.data.limit(6).show()