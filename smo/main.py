from smo import SMO
import numpy as np
from dataset import Dataset


dt = Dataset(size=400, n=3)
dt.create_explicit_dataset()
dt.export_to_xlsx()
