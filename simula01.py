import numpy as np
import matplotlib.pyplot as plt

# Parámetros del medio turbio (ejemplo: tejido biológico)
mu_a = 0.1  # Coeficiente de absorción (mm^-1)
mu_s = 100.0  # Coeficiente de scattering (mm^-1)
g = 0.9      # Factor de anisotropía
mu_s_prime = mu_s * (1 - g)  # Scattering reducido

# Coeficiente de difusión
D = 1 / (3 * (mu_a + mu_s_prime))

# Coeficiente de atenuación efectiva
mu_eff = np.sqrt(mu_a / D)

# Potencia de la fuente (mW)
P0 = 1

# Rango de profundidades (eje -z)
z = np.linspace(0.01, 10, 100)  # desde 0.01 mm hasta 10 mm

# Cálculo de la fluencia Φ(z)
fluence = (P0 / (4 * np.pi * D)) * (np.exp(-mu_eff * z) / z)

# Gráfico
plt.figure(figsize=(8, 6))
plt.plot(z, fluence, 'r-', linewidth=2)
plt.title('Distribución de Fluencia en el Medio Turbio', fontsize=14)
plt.xlabel('Profundidad (mm)', fontsize=12)
plt.ylabel('Fluencia (mW/mm²)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
#plt.yscale('log')  # Escala logarítmica para mejor visualización
plt.show()