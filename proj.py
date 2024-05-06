import numpy as np
import matplotlib.pyplot as plt

def sir_immunity_step(S, I, R, beta, gamma, delta, dt):
    dS_dt = -beta * S * I + delta * R
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I - delta * R
    S_new = S + dS_dt * dt
    I_new = I + dI_dt * dt
    R_new = R + dR_dt * dt
    return S_new, I_new, R_new

def simulate_sir_immunity(S0, I0, R0, beta, gamma, delta, T, dt):
    t_values = np.arange(0, T, dt)
    S_values, I_values, R_values = [S0], [I0], [R0]
    for t in t_values[1:]:
        S, I, R = S_values[-1], I_values[-1], R_values[-1]
        S_new, I_new, R_new = sir_immunity_step(S, I, R, beta, gamma, delta, dt)
        S_values.append(S_new)
        I_values.append(I_new)
        R_values.append(R_new)
    return t_values, S_values, I_values, R_values

def main():
    S0 = 990    
    I0 = 10     
    R0 = 0      
    beta = 0.3  
    gamma = 0.1 
    delta = 0.05 
    T = 200     
    dt = 0.1    

    t, S, I, R = simulate_sir_immunity(S0, I0, R0, beta, gamma, delta, T, dt)

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infectious')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model Dynamics with Time-Limited Immunity')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
