import numpy as np
import matplotlib.pyplot as plt

# Parámetros del medio
mu_a = 0.1
mu_s = 10.0
mu_t = mu_a + mu_s
g = 0.9
n_photons = 10000

# Histograma de absorción (profundidad)
depth = np.linspace(0, 5, 100)
absorption_profile = np.zeros_like(depth)

# Función para generar ángulos de dispersión según Henyey-Greenstein
def sample_hg(g):
    if g == 0:
        return 2*np.random.rand() - 1
    else:
        s = 2*np.random.rand() - 1
        return (1/(2*g)) * (1 + g**2 - ((1 - g**2)/(1 - g + 2*g*np.random.rand()))**2)

for _ in range(n_photons):
    z = 0
    weight = 1.0
    while weight > 0.01:
        s = -np.log(np.random.rand()) / mu_t
        z += s

        absorb = weight * (mu_a / mu_t)
        weight -= absorb

        idx = int((z / 5) * len(depth))
        if 0 <= idx < len(depth):
            absorption_profile[idx] += absorb

        cos_theta = sample_hg(g)
        # (para esta simulación 1D ignoramos cambios en dirección lateral)

# Graficar perfil de absorción
plt.plot(depth, absorption_profile)
plt.xlabel('Profundidad z [mm]')
plt.ylabel('Absorción acumulada')
plt.title('Perfil de absorción (Monte Carlo)')
plt.grid(True)
plt.yscale('log')
plt.show()