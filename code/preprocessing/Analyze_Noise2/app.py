from NoiseManager import NoiseManager

def ncep_fnl():
    print("Processing NCEP-FNL")
    noiseManager = NoiseManager(T="ncep-fnl")
    noiseManager.reset()
    noiseManager.loadRawNoises("../data/analyze/fnl/noise.csv")
    noiseManager.processRawNoises(6)
    noiseManager.exportToCsv("../data/analyze/fnl/noise_lap.csv")
    print("Done.")

def nasa_merra2():
    print("Processing NASA-MERRA2")
    noiseManager = NoiseManager(T="nasa-merra2")
    noiseManager.reset()
    noiseManager.loadRawNoises("../data/analyze/merra2/noise.csv")
    noiseManager.processRawNoises(3)
    noiseManager.exportToCsv("../data/analyze/merra2/noise_lap.csv")
    print("Done.")

if __name__ == "__main__":
    print("Noise2")
    ncep_fnl()
    nasa_merra2()
    print("Done.")
    