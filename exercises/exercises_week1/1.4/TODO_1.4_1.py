import numpy as np
import matplotlib.pyplot as plt

u_thresh = -50 * 1e-3  # milli Volt
u_rest = -65 * 1e-3  # milli Volt
delta_t = 1e-5


# Point 1.1
def LIF(um_0, I, T):
    Rm = 10 * 1e6  # Mega Ohm
    Cm = 1 * 1e-9  # Nano Farad

    um_t = np.zeros((int(T // delta_t)))
    um_t[0] = um_0

    dum_dt = lambda um_t: (u_rest - um_t + Rm * I) / (Rm * Cm)

    # TODO Calculate the um_t from T = 0 until T in steps of delta-t
    for i in range(1, len(um_t)):
        um_t[i] = um_t[i - 1] + dum_dt(um_t[i - 1]) * delta_t
        if um_t[i] >= u_thresh:
            um_t[i] = u_rest  # Reset to resting potential after spike
    return um_t


# Point 1.2
# TODO: Calculate the membrane potential using the LIF function from Point 1.1
membrane_potential = LIF(um_0=-65e-3, I=1.6e-9, T=0.1)

plt.figure(figsize=(7, 5))
plt.plot(list(range(int(0.1 // 1e-5))), membrane_potential)
plt.show()


# Point 1.3
# TODO: Define a function to calculate the interspike intervals
def calculate_isi(um_t):
    for i in range(1, len(um_t)):
        if um_t[i] == u_rest:
            if i != 0:
                return (i - 1) * delta_t
    return 0


isi = calculate_isi(membrane_potential)
print("ISI: ", isi)
print("Spiking frequency: ", 1 / isi)

# TODO: Define a function to calculate the spiking frequency of a whole experiment
# spiking_frequency =

#membrane_potential2 = LIF(TODO)
#plt.figure(figsize=(7,5))
#plt.plot(list(range(int(0.1//1e-5))), membrane_potential2)
#plt.show()


# Point 1.4
plt.figure(figsize=(7, 5))
spikes = []
# TODO write the code to accumulate the spikes
# for current in ...: #Solution here

f = []
for i in np.arange(0, 5.5e-9, 0.5e-9):
    membrane_potential = LIF(um_0=-65e-3, I=i, T=0.1)
    isi = calculate_isi(membrane_potential)
    if isi != 0:
        f.append(1/isi) 
    else: 
        f.append(0)
    

plt.plot(list(np.arange(0, 5.5e-9, 0.5e-9)), f)
plt.xlabel("Constant current")
plt.ylabel("Spiking frequency")
plt.show()
