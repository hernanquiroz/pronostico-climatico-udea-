import numpy as np
from scipy.optimize import minimize

def simulate_delayed_oscillator(params, history, tau_months, forecast_steps):
    """
    Simula el modelo Delayed-Oscillator (Ec. 16.18, Kaper & Engler).
    dT/dt = a*T(t) - b*T(t-tau) - epsilon*T(t)^3
    """
    a, b, epsilon = params
    dt = 1.0  # Paso de tiempo de 1 mes
    n_history = len(history)
    
    # Array para almacenar historial + pronóstico
    T = np.zeros(n_history + forecast_steps)
    T[:n_history] = history
    
    for t in range(n_history, n_history + forecast_steps):
        t_current = t - 1
        t_delayed = t_current - tau_months
        
        # Si el retraso cae antes del inicio de los datos, usar el primer valor disponible
        if t_delayed < 0:
            t_delayed = 0
            
        # Ecuación en diferencias finitas (Euler hacia adelante)
        dT_dt = a * T[t_current] - b * T[t_delayed] - epsilon * (T[t_current] ** 3)
        T[t] = T[t_current] + dt * dT_dt
        
    return T

def calibrate_and_predict(oni_history, tau_months=7, forecast_steps=12):
    """
    Calibra los parámetros (a, b, epsilon) usando los últimos 240 meses (20 años)
    y genera un pronóstico a 'forecast_steps' meses.
    """
    def objective_function(params, history, tau):
        train_length = min(240, len(history))
        T_sim = simulate_delayed_oscillator(params, history[:-train_length], tau, train_length)
        T_sim_relevant = T_sim[-train_length:]
        T_real_relevant = history[-train_length:]
        return np.mean((T_sim_relevant - T_real_relevant) ** 2)

    # Valores iniciales basados en literatura (Jin 1997, Kaper & Engler)
    initial_guess = [0.15, 0.25, 0.02]
    
    # Optimización con límites para evitar soluciones no físicas
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(oni_history, tau_months), 
        bounds=[(0.0, 0.5), (0.0, 0.5), (0.0, 0.1)]
    )
    
    a_opt, b_opt, eps_opt = result.x
    
    # Generar pronóstico
    T_full = simulate_delayed_oscillator([a_opt, b_opt, eps_opt], oni_history, tau_months, forecast_steps)
    oni_forecast = T_full[len(oni_history):]
    
    return {
        'params': {'a': a_opt, 'b': b_opt, 'epsilon': eps_opt},
        'forecast': oni_forecast,
        'last_historical': oni_history[-1]
    }
