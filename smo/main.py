from smo import SMO
import numpy as np
from dataset import Dataset


dt = Dataset(size=400, n=3)
dt.create_explicit_dataset()
dt.export_to_csv()

dt = Dataset(size=100, n=3)
dt.create_explicit_dataset(with_y=False)
dt.export_to_csv("features")
