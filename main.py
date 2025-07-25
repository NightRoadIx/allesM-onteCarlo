"""
Simulación de Camino Aleatorio 2D con Visualización Mejorada
Combina matplotlib para gráficos base y seaborn para estilos avanzados
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def random_walk_2D(N, L):
    """
    Simula un camino aleatorio en 2D con pasos de longitud fija.

    Parámetros:
        N (int): Número total de pasos
        L (float): Longitud de cada paso

    Devuelve:
        tuple: Dos arrays numpy con coordenadas (x, y) de la trayectoria
               Incluyendo el punto inicial (0,0)
    """
    # Inicialización de arrays (eficiente con numpy)
    x = np.zeros(N + 1)  # +1 para incluir el origen
    y = np.zeros(N + 1)

    # Generación de ángulos aleatorios (distribución uniforme en [0, 2π])
    angles = 2 * np.pi * np.random.random(N)

    # Cálculo vectorizado de posiciones (óptimo para N grande)
    x[1:] = np.cumsum(L * np.cos(angles))  # Componente x
    y[1:] = np.cumsum(L * np.sin(angles))  # Componente y

    return x, y

# Configuración de estilos con seaborn (solo afecta al estilo visual)
sns.set(style="whitegrid",  # Fondo blanco con cuadrícula
        rc={"axes.grid": True,  # Habilitar cuadrícula
            "grid.linestyle": "--",  # Estilo de línea punteada
            "grid.alpha": 0.4})  # Transparencia de la cuadrícula

# Parámetros de simulación
N = 10  # Número de pasos
L = 1.0  # Longitud del paso (unidades arbitrarias)

# Ejecución de la simulación
x, y = random_walk_2D(N, L)
dist_final = np.sqrt(x[-1] ** 2 + y[-1] ** 2)  # Distancia euclidiana final

# Configuración de la figura
fig, ax = plt.subplots(figsize=(10, 10))

# Visualización principal con matplotlib (más estable para trayectorias)
ax.plot(x, y, color='royalblue', alpha=0.7, linewidth=0.8,
        label='Trayectoria')

# Puntos destacados (inicio/fin) con estilo mejorado
ax.scatter(0, 0, color='crimson', s=100, label='Origen (0, 0)',
           edgecolor='black', linewidth=0.8)
ax.scatter(x[-1], y[-1], color='limegreen', s=100,
           label=f'Final ({x[-1]:.1f}, {y[-1]:.1f})',
           edgecolor='black', linewidth=0.8)

# Elementos del gráfico
ax.set_title(f'Simulación de Camino Aleatorio 2D\n{N} pasos de longitud {L}',
             pad=20, fontsize=14)
ax.set_xlabel('Coordenada X', fontsize=12)
ax.set_ylabel('Coordenada Y', fontsize=12)
ax.legend(loc='upper left', framealpha=1, edgecolor='black')

# Ajustes de escala y presentación
ax.axis('equal')  # Misma escala en ambos ejes
ax.set_aspect('equal', adjustable='datalim')

# Cuadrícula configurada por seaborn
ax.grid(True, linestyle='--', alpha=0.4)

# Anotación profesional con distancia final
bbox_props = dict(boxstyle="round", facecolor="white", alpha=0.8,
                  edgecolor="gray", linewidth=0.5)
ax.text(0.02, 0.98, f'Distancia final: {dist_final:.2f} unidades',
        transform=ax.transAxes, verticalalignment='top',
        bbox=bbox_props, fontsize=10)

plt.tight_layout()
plt.show()