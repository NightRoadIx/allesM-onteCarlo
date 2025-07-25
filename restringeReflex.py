"""
Simulación de Camino Aleatorio 2D con:
- Detección precisa de colisiones en fronteras
- Marcado correcto de puntos de choque
- Visualización limpia y profesional
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Union


def random_walk_2D_with_accurate_collisions(N: int,
                                            L_range: Tuple[float, float],
                                            allowed_quadrants: Union[str, Tuple[str, ...]] = ('I', 'II', 'III', 'IV')
                                            ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """
    Versión corregida con detección precisa de colisiones

    Parámetros:
        N: Número de pasos
        L_range: Tupla (L_min, L_max) para longitud de paso
        allowed_quadrants: Cuadrantes permitidos

    Devuelve:
        tuple: (x_coords, y_coords, collision_points)
    """
    # Validación de parámetros
    allowed_quadrants = [allowed_quadrants] if isinstance(allowed_quadrants, str) else list(allowed_quadrants)
    valid_quadrants = ['I', 'II', 'III', 'IV']
    if not all(q in valid_quadrants for q in allowed_quadrants):
        raise ValueError(f"Cuadrantes deben ser alguno de: {valid_quadrants}")

    # Inicialización
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    collision_points = []
    L_min, L_max = L_range

    # Mapeo de condiciones de cuadrantes
    quadrant_conditions = {
        'I': lambda x, y: x >= 0 and y >= 0,
        'II': lambda x, y: x <= 0 and y >= 0,
        'III': lambda x, y: x <= 0 and y <= 0,
        'IV': lambda x, y: x >= 0 and y <= 0
    }

    def is_allowed(x_pos, y_pos):
        return any(q_cond(x_pos, y_pos) for q_cond in [quadrant_conditions[q] for q in allowed_quadrants])

    # Simulación con detección mejorada de colisiones
    for i in range(1, N + 1):
        angle = 2 * np.pi * np.random.random()
        L = np.random.uniform(L_min, L_max)
        dx, dy = L * np.cos(angle), L * np.sin(angle)

        x_temp, y_temp = x[i - 1] + dx, y[i - 1] + dy

        if not is_allowed(x_temp, y_temp):
            # Calcular intersección con los ejes de forma precisa
            t_x = t_y = float('inf')

            # Intersección con eje Y (x=0)
            if dx != 0:
                t_x = -x[i - 1] / dx
                y_intersect = y[i - 1] + t_x * dy

            # Intersección con eje X (y=0)
            if dy != 0:
                t_y = -y[i - 1] / dy
                x_intersect = x[i - 1] + t_y * dx

            # Seleccionar la primera intersección válida
            t = 1.0
            x_coll, y_coll = x[i - 1], y[i - 1]  # Por defecto, quedarse en posición actual

            if 0 <= t_x < t:
                t = t_x
                x_coll, y_coll = 0.0, y_intersect

            if 0 <= t_y < t:
                t = t_y
                x_coll, y_coll = x_intersect, 0.0

            # Solo registrar colisión si encontramos una intersección válida
            if t < 1.0:
                collision_points.append((x_coll, y_coll))

                # Calcular dirección reflejada
                remaining_L = L * (1 - t)
                reflect_x = abs(x_coll) < 1e-10
                reflect_y = abs(y_coll) < 1e-10

                # Vector de dirección después del rebote
                new_dx = -dx if reflect_x else dx
                new_dy = -dy if reflect_y else dy

                # Normalizar y escalar
                norm = np.hypot(new_dx, new_dy)
                if norm > 0:
                    new_dx = new_dx / norm * remaining_L
                    new_dy = new_dy / norm * remaining_L

                x_temp = x_coll + new_dx
                y_temp = y_coll + new_dy

            # Si no se encontró intersección válida, quedarse en posición actual
            else:
                x_temp, y_temp = x[i - 1], y[i - 1]

        x[i], y[i] = x_temp, y_temp

    return x, y, collision_points


# Configuración visual
sns.set(style="whitegrid", palette="husl", font_scale=1.1)
fig, ax = plt.subplots(figsize=(10, 10))

# Parámetros de simulación
N = 5000
L_range = (0.1, 1.0)
quadrants = ('I')  # Solo cuadrantes derecho

# Ejecutar simulación corregida
x, y, collisions = random_walk_2D_with_accurate_collisions(N, L_range, quadrants)

# Dibujar trayectoria
ax.plot(x, y, 'b-', alpha=0.7, linewidth=1.5, label='Trayectoria')

# Puntos de inicio/fin
ax.scatter(0, 0, color='red', s=100, label='Origen', zorder=3)
ax.scatter(x[-1], y[-1], color='green', s=100, label=f'Final ({x[-1]:.1f}, {y[-1]:.1f})', zorder=3)

# Dibujar puntos de colisión (solo si existen)
if collisions:
    coll_x, coll_y = zip(*collisions)
    ax.scatter(coll_x, coll_y, color='yellow', s=60, edgecolor='black',
               linewidth=0.8, label=f'Choques ({len(collisions)})', zorder=4)

# Líneas divisorias
ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
ax.axvline(0, color='black', linestyle='--', alpha=0.4, linewidth=0.8)

# Resaltar cuadrantes permitidos
for q in quadrants:
    if q == 'I':
        ax.fill_between([0, max(1, np.max(x))], 0, max(1, np.max(y)),
                        color='green', alpha=0.03)
    elif q == 'IV':
        ax.fill_between([0, max(1, np.max(x))], min(-1, np.min(y)), 0,
                        color='green', alpha=0.03)

# Configuración del gráfico
title = f"Camino Aleatorio con Choques\nPasos: {N} | L: {L_range[0]}-{L_range[1]} | Cuadrantes: {', '.join(quadrants)}"
ax.set_title(title, pad=15)
ax.set_xlabel("Coordenada X")
ax.set_ylabel("Coordenada Y")
ax.legend(loc='upper left')
ax.axis('equal')
ax.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.show()