"""
Simulación de Camino Aleatorio 3D con:
- Paso fijo L=1
- Ángulos cenital (θ) y acimutal (φ) aleatorios uniformes
- Visualización interactiva 3D
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def random_walk_3D(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera un camino aleatorio 3D con paso unitario

    Parámetros:
        N: Número de pasos

    Devuelve:
        tuple: (x, y, z) arrays con las coordenadas de la trayectoria
    """
    # Inicializar arrays (incluyendo punto de partida)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    z = np.zeros(N + 1)

    # Generar ángulos aleatorios con distribución uniforme en esfera:
    # θ ∈ [0, π] (cenital)
    # φ ∈ [0, 2π] (acimutal)
    theta = np.arccos(1 - 2 * np.random.random(N))  # Distribución uniforme en cosθ
    phi = 2 * np.pi * np.random.random(N)

    # Calcular componentes del paso unitario
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)

    # Posiciones acumuladas
    x[1:] = np.cumsum(dx)
    y[1:] = np.cumsum(dy)
    z[1:] = np.cumsum(dz)

    return x, y, z


# Configuración estética
sns.set(style="whitegrid", palette="husl", font_scale=1.1)

# Parámetros de simulación
N = 800  # Número de pasos

# Generar trayectoria
x, y, z = random_walk_3D(N)

# Visualización 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Dibujar trayectoria con degradado de color
ax.plot(x, y, z, alpha=0.8, linewidth=0.8,
        color='royalblue', label='Trayectoria')

# Puntos de inicio y fin
ax.scatter([0], [0], [0], color='red', s=100,
           label='Origen (0,0,0)', depthshade=False)
ax.scatter([x[-1]], [y[-1]], [z[-1]], color='green', s=100,
           label=f'Final ({x[-1]:.1f}, {y[-1]:.1f}, {z[-1]:.1f})',
           depthshade=False)

# Configuración del gráfico
ax.set_title(f'Camino Aleatorio 3D\n{N} pasos unitarios', pad=20, fontsize=14)
ax.set_xlabel('Eje X', fontsize=12)
ax.set_ylabel('Eje Y', fontsize=12)
ax.set_zlabel('Eje Z', fontsize=12)
ax.legend(loc='upper left', fontsize=10)

# Ajustar límites para mejor visualización
max_range = np.max(np.abs(np.concatenate([x, y, z]))) * 1.1
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

# Cuadrícula y ángulo de vista
ax.grid(True, linestyle=':', alpha=0.4)
ax.xaxis.set_pane_color((0.98, 0.98, 0.98, 0.8))
ax.yaxis.set_pane_color((0.98, 0.98, 0.98, 0.8))
ax.zaxis.set_pane_color((0.98, 0.98, 0.98, 0.8))
ax.view_init(elev=25, azim=45)  # Ángulo de vista inicial

plt.tight_layout()
plt.show()