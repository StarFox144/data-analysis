import numpy as np
from skfuzzy.membership import trimf

# --- SAFE TRIMF ДЛЯ СКАЛЯРІВ ---
def trimf_scalar(x, params):
    x_arr = np.array([x])           # робимо масив
    y = trimf(x_arr, params)        # обробка
    return float(y[0])              # повертаємо число


# --- ФУНКЦІЇ НАЛЕЖНОСТІ x1 ---
def x1_low(x):  return trimf_scalar(x, [-7, -7, -2])
def x1_med(x):  return trimf_scalar(x, [-5, -2,  2])
def x1_high(x): return trimf_scalar(x, [ 0,  3,  3])

# --- ФУНКЦІЇ НАЛЕЖНОСТІ x2 ---
def x2_low(x):  return trimf_scalar(x, [-4.4, -4.4, -1])
def x2_med(x):  return trimf_scalar(x, [-3,   -1,   1])
def x2_high(x): return trimf_scalar(x, [ 0,   1.7,  1.7])


# --- НЕЧІТКИЙ ВИВІД SUGENO ---
def sugeno_output(x1, x2):
    # активації правил
    r1 = x1_med(x1)
    r2 = min(x1_high(x1), x2_high(x2))
    r3 = min(x1_high(x1), x2_low(x2))
    r4 = min(x1_low(x1),  x2_med(x2))
    r5 = min(x1_low(x1),  x2_low(x2))
    r6 = min(x1_low(x1),  x2_high(x2))

    # виходи правил
    z1 = 0
    z2 = 2*x1 + 2*x2 + 1
    z3 = 4*x1 - x2
    z4 = 8*x1 + 2*x2 + 8
    z5 = 50
    z6 = 50

    # Sugeno дефазифікація
    num = r1*z1 + r2*z2 + r3*z3 + r4*z4 + r5*z5 + r6*z6
    den = r1 + r2 + r3 + r4 + r5 + r6

    return num / den if den != 0 else 0


# --- ТЕСТ ---
if __name__ == "__main__":
    x1_val = -3
    x2_val = 0.5

    out = sugeno_output(x1_val, x2_val)
    print("Sugeno output =", out)
