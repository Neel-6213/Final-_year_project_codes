#alpha_m and theta _n are randomnly chosen uniformly 
import numpy as np
import matplotlib.pyplot as plt

wavelength = 3e8/28e9
N_RIS = 64
N_BS = 4
d_RIS = wavelength / 2
d_BS = wavelength / 2

theta_RB = 45 # Angle of Arrival at RIS from BS
theta_BR = 30 # Angle of Departure from BS to RIS

transmit_power = -10 # transmit power(dbm)
noise_power = -80 # Noise power (dbm)
path_loss=-90
time_slots = 1000    # Number of sweeps per K 
K_array = [10,100,1000,10000]


def steering_vector(angle_deg, N, d):
    angle_rad = np.deg2rad(angle_deg)
    n = np.arange(N).reshape(-1, 1)
    return np.exp(1j * 2 * np.pi * n * d * np.sin(angle_rad) / wavelength)

a_N_arrival = steering_vector(theta_RB, N_RIS, d_RIS)
a_M_depart = steering_vector(theta_BR, N_BS, d_BS)
H_BR = a_N_arrival @ a_M_depart.conj().T

m_array = np.arange(N_BS).reshape(-1, 1)
alpha_m = (2 * np.pi / wavelength) * m_array * d_BS * np.sin(np.deg2rad(theta_BR))#randomize phase shifts
w_BS = np.exp(1j * alpha_m)


baseline_rates = []
protocol_b_rates = []

for K in K_array:
    # Place K users randomly in the sector [-60, 60]
    ue_angles = np.random.uniform(0, 180, K)

    # --- BASELINE (Perfect CSI) ---

    h_Rk_ideal = steering_vector(ue_angles[0], N_RIS, d_RIS)
    
 
    theta_n_ideal = (2 * np.pi / wavelength) * np.arange(N_RIS).reshape(-1,1)* d_RIS * (np.sin(np.deg2rad(ue_angles[0])) - np.sin(np.deg2rad(theta_RB)))
    perfect_RIS = np.diag(np.exp(1j * theta_n_ideal).flatten())
    
    H_eff_ideal = h_Rk_ideal.conj().T @ perfect_RIS @ H_BR @ w_BS
    snr_ideal = (10**(transmit_power/10)*(10**(path_loss/10)) * np.abs(H_eff_ideal[0, 0])**2)/(10**(noise_power/10))
    baseline_rates.append(np.log2(1 + snr_ideal))

  

    tau =  50 # The moving average memory window
    T_k = np.ones(K) * 0.1 

    steady_state_rates = [] # Only record rates after the memory stabilizes

    for t in range(time_slots):
        #  Blindly guess an angle phi
        phi_guess = np.random.uniform(0, 5)#0-30 narrow
        
        # Configure the RIS based on the guess
        theta_n = (2 * np.pi / wavelength) * np.arange(N_RIS).reshape(-1,1) * d_RIS * (np.sin(np.deg2rad(phi_guess)) - np.sin(np.deg2rad(theta_RB)))
        reflection_matrix = np.diag(np.exp(1j * theta_n).flatten())
        
        # Calculate Instantaneous Rates for all K users
        instantaneous_rates = np.zeros(K)
        for i, angle in enumerate(ue_angles):
            h_Rk = steering_vector(angle, N_RIS, d_RIS)
            alpha_m1=(2 * np.pi / wavelength) * m_array * d_BS * np.sin(np.deg2rad(theta_BR+np.random.uniform(0,180)))
            w_BS1=np.exp(1j * alpha_m1)
            H_eff = h_Rk.conj().T @ reflection_matrix @ H_BR @ w_BS1
        
            snr = (10**(transmit_power/10)*10**(path_loss/10) * np.abs(H_eff[0, 0])**2)/(10**(noise_power/10))
            instantaneous_rates[i] = np.log2(1 + snr)
            
        #  Proportional Fair Scheduler
        pf_metrics = instantaneous_rates / T_k
        winner = np.argmax(pf_metrics) #
        
        if t > tau:
            steady_state_rates.append(instantaneous_rates[winner])

        for i in range(K):
            if i == winner:
                T_k[i] = (1 - 1/tau) * T_k[i] + (1/tau) * instantaneous_rates[winner]
            else:
                T_k[i] = (1 - 1/tau) * T_k[i]

    protocol_b_rates.append(np.mean(steady_state_rates))
    print(f"Users (K) = {K:2d} | Baseline = {baseline_rates[-1]:.2f} bps/Hz | Protocol B (PF) = {protocol_b_rates[-1]:.2f} bps/Hz")
print("done")