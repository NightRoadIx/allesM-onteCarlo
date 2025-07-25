"""
Simulación de Fuentes de Luz con 3 opciones:
1. Fuente colimada uniforme
2. Fuente divergente uniforme
3. Fuente gaussiana (como láser/diodo)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import Circle


def generar_fotones(N: int, radio: float, tipo_fuente: str, sigma_gauss: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera fotones según el tipo de fuente especificado

    Parámetros:
        N: Número de fotones
        radio: Radio de la fuente (mm)
        tipo_fuente: 'colimada', 'divergente' o 'gaussiana'
        sigma_gauss: Ancho del perfil gaussiano (solo para tipo 'gaussiana')

    Devuelve:
        tuple: (x, y) coordenadas iniciales de los fotones
    """
    theta = 2 * np.pi * np.random.random(N)

    if tipo_fuente in ['colimada', 'divergente']:
        # Distribución uniforme en círculo
        r = radio * np.sqrt(np.random.random(N))
    elif tipo_fuente == 'gaussiana':
        # Distribución gaussiana radial (intensidad ∝ exp(-r²/(2σ²)))
        r = np.abs(np.random.normal(0, sigma_gauss, N))
        r = np.clip(r, 0, radio)  # Recortar al radio máximo
    else:
        raise ValueError("Tipo de fuente debe ser 'colimada', 'divergente' o 'gaussiana'")

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def propagar_fotones(x_inicial: np.ndarray, y_inicial: np.ndarray,
                     distancia: float, tipo_fuente: str,
                     sigma_divergencia: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Propaga fotones según el tipo de fuente

    Parámetros:
        x_inicial, y_inicial: Posiciones iniciales
        distancia: Distancia a la superficie (mm)
        tipo_fuente: 'colimada', 'divergente' o 'gaussiana'
        sigma_divergencia: Para fuente divergente (radianes)

    Devuelve:
        tuple: (x_final, y_final) posiciones en la superficie
    """
    if tipo_fuente == 'colimada':
        return x_inicial.copy(), y_inicial.copy()

    elif tipo_fuente == 'divergente':
        theta_x = np.random.normal(0, sigma_divergencia, len(x_inicial))
        theta_y = np.random.normal(0, sigma_divergencia, len(x_inicial))
        return x_inicial + distancia * np.tan(theta_x), y_inicial + distancia * np.tan(theta_y)

    elif tipo_fuente == 'gaussiana':
        # Para láser: pequeña divergencia + perfil gaussiano
        theta_r = np.abs(np.random.normal(0, sigma_divergencia, len(x_inicial)))
        phi = 2 * np.pi * np.random.random(len(x_inicial))

        theta_x = theta_r * np.cos(phi)
        theta_y = theta_r * np.sin(phi)

        return x_inicial + distancia * np.tan(theta_x), y_inicial + distancia * np.tan(theta_y)

    else:
        raise ValueError("Tipo de fuente no válido")


# Parámetros de simulación
N_fotones = 100000
radio_fuente = 1.0  # mm
distancia = 10.0  # mm
tipo_fuente = 'gaussiana'  # Opciones: 'colimada', 'divergente', 'gaussiana'

# Parámetros específicos por tipo
sigma_divergencia = 0.3  # radianes (para divergente y gaussiana)
sigma_gauss = 0.9 # Ancho gaussiano (solo para 'gaussiana')

# Generar fotones
x_inicial, y_inicial = generar_fotones(N_fotones, radio_fuente, tipo_fuente, sigma_gauss)

# Propagación
x_final, y_final = propagar_fotones(x_inicial, y_inicial, distancia, tipo_fuente, sigma_divergencia)

# Configuración visual
sns.set(style="whitegrid", font_scale=1.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# --- Gráfico 1: Distribución inicial ---
ax1.hist2d(x_inicial, y_inicial, bins=100, cmap='inferno', norm=PowerNorm(gamma=0.5))
ax1.add_patch(Circle((0, 0), radio_fuente, color='cyan', fill=False, linestyle='--', linewidth=2))
ax1.set_title(f'Distribución Inicial en la Fuente\n(Tipo: {tipo_fuente})', pad=15)
ax1.set_xlabel('Posición X (mm)')
ax1.set_ylabel('Posición Y (mm)')
ax1.axis('equal')

# --- Gráfico 2: Distribución en superficie ---
hb = ax2.hexbin(x_final, y_final, gridsize=100, cmap='inferno',
                norm=LogNorm(), mincnt=1, edgecolor='none')
cbar = fig.colorbar(hb, ax=ax2, label='Densidad de fotones (log)')
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

ax2.add_patch(Circle((0, 0), radio_fuente, color='cyan', fill=False, linestyle='--', linewidth=2, label='Radio fuente'))
ax2.set_title(f'Distribución en Superficie a {distancia} mm\n'
              f'Fotones: {N_fotones:,} | ' +
              (f'σ_gauss: {sigma_gauss:.2f} mm | ' if tipo_fuente == 'gaussiana' else '') +
              (f'σ_div: {sigma_divergencia:.2f} rad' if tipo_fuente != 'colimada' else 'Colimación perfecta'),
              pad=15)
ax2.set_xlabel('Posición X (mm)')
ax2.set_ylabel('Posición Y (mm)')
ax2.legend()
ax2.axis('equal')

plt.tight_layout()
plt.show()