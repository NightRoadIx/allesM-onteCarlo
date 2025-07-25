import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns

# Configuración inicial
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.figsize'] = [14, 8]


def generar_fotones(N, tipo, **kwargs):
    """Genera posiciones iniciales de fotones para el tipo seleccionado"""
    theta = 2 * np.pi * np.random.random(N)
    r = np.zeros(N)  # Inicialización para todos los casos

    if tipo == 'lápiz':
        r = 0.01 * kwargs.get('radio', 1) * np.ones(N)

    elif tipo == 'colimada':
        r = kwargs.get('radio', 1) * np.sqrt(np.random.random(N))

    elif tipo == 'divergente':
        r = kwargs.get('radio', 1) * np.sqrt(np.random.random(N))

    elif tipo == 'gaussiana':
        sigma = kwargs.get('sigma', 0.3)
        r = np.abs(np.random.normal(0, sigma, N))
        r = np.clip(r, 0, kwargs.get('radio', 1))

    elif tipo == 'lambertiana':
        r = kwargs.get('radio', 1) * np.sqrt(np.random.random(N))
        global theta_global
        theta_global = np.arcsin(np.sqrt(np.random.random(N)))

    elif tipo == 'plana_bordes_difusos':
        M = int(N * 1.5)
        r_temp = kwargs.get('radio', 1) * np.random.random(M)
        perfil = np.exp(-(r_temp ** 10) / (2 * (kwargs.get('radio', 1) * 0.7) ** 10))
        mask = np.random.random(M) < perfil
        r = r_temp[mask][:N]
        theta = theta[:len(r)]
    else:
        raise ValueError(f"Tipo de fuente desconocido: {tipo}")

    # Cálculo de coordenadas
    x = r * np.cos(theta[:len(r)])
    y = r * np.sin(theta[:len(r)])
    return x, y


def propagar_fotones(x, y, distancia, tipo, **kwargs):
    """Propaga los fotones según el tipo de fuente"""
    if tipo in ['lápiz', 'colimada']:
        return x.copy(), y.copy()

    if tipo == 'divergente':
        theta = np.random.normal(0, kwargs.get('sigma_div', 0.1), len(x))

    elif tipo == 'gaussiana':
        theta = np.abs(np.random.normal(0, kwargs.get('sigma_div', 0.05), len(x)))

    elif tipo == 'lambertiana':
        theta = theta_global[:len(x)]

    elif tipo == 'plana_bordes_difusos':
        theta = np.random.normal(0, kwargs.get('sigma_div', 0.08), len(x))
    else:
        raise ValueError(f"Tipo de fuente desconocido: {tipo}")

    phi = 2 * np.pi * np.random.random(len(x))
    return (x + distancia * np.tan(theta) * np.cos(phi),
            y + distancia * np.tan(theta) * np.sin(phi))


# Parámetros de simulación
N = 50000
radio = 1.0
distancia = 10.0

# Selección interactiva
print("Selecciona el tipo de fuente a simular:")
print("1. Haz tipo lápiz")
print("2. Fuente colimada")
print("3. Fuente divergente")
print("4. Fuente gaussiana")
print("5. Fuente lambertiana")
print("6. Fuente plana con bordes difusos")

opcion = int(input("Ingresa el número correspondiente (1-6): "))

fuentes = [
    ('lápiz', {'sigma_div': 0.01}),
    ('colimada', {}),
    ('divergente', {'sigma_div': 0.15}),
    ('gaussiana', {'sigma': 0.4, 'sigma_div': 0.05}),
    ('lambertiana', {}),
    ('plana_bordes_difusos', {'sigma_div': 0.1})
]

nombre, params = fuentes[opcion - 1]

# Simulación
try:
    x_inicial, y_inicial = generar_fotones(N, tipo=nombre, radio=radio, **params)
    x_final, y_final = propagar_fotones(x_inicial, y_inicial, distancia, tipo=nombre, **params)

    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Gráfico inicial
    hb1 = ax1.hexbin(x_inicial, y_inicial, gridsize=50, cmap='inferno', norm=PowerNorm(gamma=0.5))
    ax1.set_title(f'Distribución Inicial: {nombre.capitalize()}')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.axis('equal')
    plt.colorbar(hb1, ax=ax1, label='Densidad')

    # Gráfico en superficie
    hb2 = ax2.hexbin(x_final, y_final, gridsize=50, cmap='inferno', norm=PowerNorm(gamma=0.5))
    ax2.set_title(f'Distribución en Superficie (Distancia: {distancia} mm)')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.axis('equal')
    plt.colorbar(hb2, ax=ax2, label='Densidad')

    plt.suptitle(f'Simulación de Fuente de Luz: {nombre.capitalize()}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nError: {str(e)}")
    print("Posibles causas:")
    print("1. Parámetros inválidos")
    print("2. Tipo de fuente no reconocido")
    print("3. Problema con los valores de entrada\n")