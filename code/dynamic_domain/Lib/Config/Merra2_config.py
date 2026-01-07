SINGLE_VAR = ['PHIS', 'PS', 'SLP',]
PRESS_VAR = ['H', 'OMEGA', 'QI', 'QL', 'QV', 'RH', 'T', 'U', 'V']
LEVEL = 25

INP_CHANNELS = len(SINGLE_VAR) + LEVEL * len(PRESS_VAR)
PRESS_LEVEL = [1000, 975, 950, 925, 900, 875, 850, 825, 
               800, 775, 750, 725, 700, 650, 600, 550, 
               500, 450, 400, 350, 300, 250, 200, 150, 100]

LIST_VAR = [var + '_0' for var in SINGLE_VAR]
LIST_VAR.extend([var + '_' + str(level) for var in PRESS_VAR for level in PRESS_LEVEL])

methods = ['min', 'max', 'mean']