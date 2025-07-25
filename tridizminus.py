"""
Simulación de Camino Aleatorio 3D con:
- Movimiento restringido a dirección Z negativa
- Longitud de paso aleatoria uniforme en [a, b]
- Corrección del error en plot_surface
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def random_walk_3D_negative_z(N: int, L_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera camino aleatorio 3D solo en dirección Z negativa

    Parámetros:
        N: Número de pasos
        L_range: Tupla (a, b) con rango de longitudes de paso

    Devuelve:
        tuple: (x, y, z) arrays con las coordenadas de la trayectoria
    """
    # Inicializar arrays (incluyendo punto de partida)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    z = np.zeros(N + 1)

    # Generar ángulos aleatorios restringidos a Z negativa:
    theta = np.pi / 2 + np.random.random(N) * np.pi / 2  # Uniforme en [π/2, π]
    phi = 2 * np.pi * np.random.random(N)  # Uniforme en [0, 2π]

    # Longitudes de paso aleatorias
    L = np.random.uniform(L_range[0], L_range[1], N)

    # Calcular componentes del paso (z siempre negativo)
    dx = L * np.sin(theta) * np.cos(phi)
    dy = L * np.sin(theta) * np.sin(phi)
    dz = -L * np.abs(np.cos(theta))

    # Posiciones acumuladas
    x[1:] = np.cumsum(dx)
    y[1:] = np.cumsum(dy)
    z[1:] = np.cumsum(dz)

    return x, y, z


# Configuración estética
sns.set(style="whitegrid", palette="husl", font_scale=1.1)

# Parámetros de simulación
N = 5000 # Número de pasos
L_range = (0.5, 2.0)  # Rango de longitudes de paso

# Generar trayectoria
x, y, z = random_walk_3D_negative_z(N, L_range)

# Visualización 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Dibujar trayectoria con color por profundidad
for i in range(N):
    ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2],
            color=plt.cm.viridis(i / N), alpha=0.8, linewidth=1.5)

# Puntos de inicio y fin
ax.scatter([0], [0], [0], color='red', s=100, label='Origen (0,0,0)', depthshade=False)
ax.scatter([x[-1]], [y[-1]], [z[-1]], color='green', s=100,
           label=f'Final ({x[-1]:.1f}, {y[-1]:.1f}, {z[-1]:.1f})',
           depthshade=False)

# Configuración del gráfico
title = f'Camino Aleatorio 3D (Z negativa)\n{N} pasos | L ∈ [{L_range[0]}, {L_range[1]}]'
ax.set_title(title, pad=20, fontsize=14)
ax.set_xlabel('Eje X', fontsize=12)
ax.set_ylabel('Eje Y', fontsize=12)
ax.set_zlabel('Eje Z', fontsize=12)
ax.legend(loc='upper left', fontsize=10)

# Ajustar límites
max_xy = np.max(np.abs(np.concatenate([x, y]))) * 1.2
ax.set_xlim([-max_xy, max_xy])
ax.set_ylim([-max_xy, max_xy])
ax.set_zlim([np.min(z) * 1.1, max(0.5, -np.min(z) * 0.1)])

# Cuadrícula y ángulo de vista
ax.grid(True, linestyle=':', alpha=0.4)
ax.view_init(elev=30, azim=120)

# CORRECCIÓN: Plano de referencia 2D correctamente dimensionado
xx, yy = np.meshgrid(np.linspace(-max_xy, max_xy, 2), np.linspace(-max_xy, max_xy, 2))
zz = np.zeros((2, 2))  # Ahora es 2-dimensional
ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

plt.tight_layout()
plt.show()