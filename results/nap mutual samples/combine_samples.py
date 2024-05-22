import numpy as np 
# 8_0.28_layeredresults_samples_nap_10000.npy
# 8_0.28_layeredresults_samples_nap_10000_secondbatch.npy

if __name__ == "__main__":
    for q in range(4,18,2):
        for p in [.22,.24,.26,.28]:
            samples = np.load(f"{q}_{p}_layeredresults_samples_nap_10000.npy") 

            print(samples.shape)
            samples = np.append(samples, np.load(f"{q}_{p}_layeredresults_samples_nap_10000_secondbatch.npy"))

            np.save(f"{q}_{p}_layeredresults_samples_nap_10000",samples, )
    
