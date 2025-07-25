"""
Simulación de Camino Aleatorio 2D con:
- Longitud de paso aleatoria (distribución uniforme)
- Restricciones por cuadrantes cartesianos
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple


def random_walk_2D(N: int, L_range: Tuple[float, float],
                   allowed_quadrants: Union[str, Tuple[str, ...]] = ('I', 'II', 'III', 'IV')) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Simula un camino aleatorio 2D con restricciones espaciales

    Parámetros:
        N (int): Número total de pasos
        L_range (tuple): (L_min, L_max) para longitud aleatoria del paso
        allowed_quadrants: Cuadrantes permitidos ('I','II','III','IV' en cualquier combinación)

    Devuelve:
        tuple: Arrays (x, y) con la trayectoria válida
    """
    # Validación de parámetros
    if not all(q in ['I', 'II', 'III', 'IV'] for q in allowed_quadrants):
        raise ValueError("Los cuadrantes deben ser 'I','II','III' o 'IV'")

    # Inicialización
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    L_min, L_max = L_range

    # Máscaras de cuadrantes (para verificación eficiente)
    quadrant_masks = {
        'I': lambda x, y: (x >= 0) & (y >= 0),
        'II': lambda x, y: (x <= 0) & (y >= 0),
        'III': lambda x, y: (x <= 0) & (y <= 0),
        'IV': lambda x, y: (x >= 0) & (y <= 0)
    }

    # Generación de pasos con restricciones
    for i in range(1, N + 1):
        while True:
            # Paso aleatorio
            angle = 2 * np.pi * np.random.random()
            L = np.random.uniform(L_min, L_max)
            dx, dy = L * np.cos(angle), L * np.sin(angle)

            # Posición tentativa
            x_temp, y_temp = x[i - 1] + dx, y[i - 1] + dy

            # Verificación de cuadrantes permitidos
            valid = False
            for q in allowed_quadrants:
                if quadrant_masks[q](x_temp, y_temp):
                    valid = True
                    break
            if valid:
                x[i], y[i] = x_temp, y_temp
                break

    return x, y


# Configuración de estilos
sns.set(style="whitegrid", font_scale=1.1)
palette = sns.color_palette("husl", 4)

# Parámetros de simulación
N = 50  # Número de pasos
L_range = (0.5, 1.5)
quadrants = ('I', 'II')

# Ejecución
x, y = random_walk_2D(N, L_range, quadrants)

# Visualización
fig, ax = plt.subplots(figsize=(10, 10))

# Líneas divisorias de cuadrantes
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

# Trayectoria
ax.plot(x, y, color=palette[0], alpha=0.7, linewidth=1.2)

# Puntos destacados
ax.scatter(0, 0, color='red', s=100, label='Origen', zorder=3)
ax.scatter(x[-1], y[-1], color='green', s=100,
           label=f'Final ({x[-1]:.1f}, {y[-1]:.1f})', zorder=3)

# Configuración del gráfico
title = f"Camino Aleatorio 2D\nPasos: {N} | L: {L_range[0]}-{L_range[1]} | Cuadrantes: {', '.join(quadrants)}"
ax.set_title(title, pad=20)
ax.set_xlabel("Coordenada X")
ax.set_ylabel("Coordenada Y")
ax.legend()
ax.axis('equal')

# Resaltar cuadrantes permitidos
for q in quadrants:
    if q == 'I':
        ax.fill_between([0, max(1, np.max(x))], 0, max(1, np.max(y)), color='green', alpha=0.05)
    elif q == 'II':
        ax.fill_between([min(-1, np.min(x)), 0], 0, max(1, np.max(y)), color='green', alpha=0.05)
    elif q == 'III':
        ax.fill_between([min(-1, np.min(x)), 0], min(-1, np.min(y)), 0, color='green', alpha=0.05)
    elif q == 'IV':
        ax.fill_between([0, max(1, np.max(x))], min(-1, np.min(y)), 0, color='green', alpha=0.05)

plt.tight_layout()
plt.show()