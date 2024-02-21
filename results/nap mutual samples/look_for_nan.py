
import numpy as np 
import matplotlib.pyplot as plt
import sys
import distinctipy
sys.path.insert(0, '../..')
from MILT_mutual_information import *


"""
This script explores why I am getting those np.nan points. 
"""

if __name__ == "__main__":

  n_ap = 1000
  qubits = [4,6,8,10,12,14,16]
  n_layers = 60
  probs = [0,.05,.1,.2,.3,.5,.7,.9]

  for examined_layer in [4,-1]:

      for i, n_qubits in enumerate(qubits):

          results_for_each_p = []
          er_each_p = []

          for j, p in enumerate(probs):
              # Load your data based on n_qubits and p
              if p in [0, .05, .1] and n_qubits in [12,14,16]:
                  data = np.load(f"{n_qubits}_{p}_layeredresults_samples_nap_10000.npy")
              elif p in [.2,.3,.5] and n_qubits == 16:
                  data = np.load(f"{n_qubits}_{p}_layeredresults_samples_nap_10000.npy")
              elif n_qubits == 10:
                  data = np.load(f"{n_qubits}_{p}_layeredresults_samples_nap_10000.npy")
              elif p == 0:
                  continue
              elif n_qubits == 12 or n_qubits == 14:
                  data = np.load(f"{n_qubits}_{p}_layeredresults_samples_changeboth_1000.npy")
              else:
                  data = np.load(f"{n_qubits}_{p}_layeredresults_samples_changeboth.npy")

              # print("original sample data and shapes")
              # print(data)
              # print(data.shape)

              # Step 1: Get a boolean array where True represents NaN
              nan_mask = np.isnan(data)

              # Step 2: Find rows (in the 10000) that contain at least one NaN across (2, 60)
              # We collapse across the last two dimensions
              rows_with_nan = nan_mask.any(axis=-1).any(axis=-1)

              # Step 3: Index the original array to get only rows (first dimension) with NaN
              rows_containing_nan = data[rows_with_nan]
            
              print(f"Number of rows containing at least one NaN: {rows_containing_nan.shape[0]}")
              # If you want to see which rows have NaNs:
              print(f"Indices of rows containing at least one NaN: {np.where(rows_with_nan)[0]}") 
              # print(rows_containing_nan)

              # print("data after finding average mutual information")
              # data = mutual_info_changeall(data)

              # print(data.shape)
              # print(data)