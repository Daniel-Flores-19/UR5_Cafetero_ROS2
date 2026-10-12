# Pintor UR5 / UR5e — ROS 2 Humble

Un brazo UR5/UR5e dibuja con plumón sobre una hoja A4: texto (tipografía
UTEC, Ladislav Light) o imágenes tipo *line art* (logos, dibujos animados).
Todo se lanza desde **un solo launch** que abre, según el argumento `modo`:

| `modo:=sim` | `modo:=real` |
|---|---|
| Gazebo + RViz (en la GPU NVIDIA) | `ur_robot_driver` + RViz |
| Controladores de Gazebo (`ur_simulation_gazebo`) | Controladores del robot real |
| **Sin calibración**: dibuja a una Z plana igual a la de la última calibración guardada | **Calibración de superficie 4x4** (16 puntos con el sensor de fuerza) y dibujo siguiendo z(x, y) |

```bash
source ~/pintor_ws/install/setup.bash
ros2 launch ur5_algoritmos pintor.launch.py --show-args   # todos los argumentos
```

---

## Índice

1. [Novedades](#1-novedades)
2. [Requisitos y compilación](#2-requisitos-y-compilación)
3. [Cómo funciona el launch](#3-cómo-funciona-el-launch)
4. [Ejemplos de uso](#4-ejemplos-de-uso)
5. [Argumentos del launch](#5-argumentos-del-launch)
6. [Robot real: consideraciones y puesta en marcha](#6-robot-real-consideraciones-y-puesta-en-marcha)
7. [Notas de la simulación](#7-notas-de-la-simulación)
8. [Generación de trayectorias (Fourier)](#8-generación-de-trayectorias-fourier)
9. [Archivos del proyecto](#9-archivos-del-proyecto)
10. [Problemas frecuentes](#10-problemas-frecuentes)
11. [Pendientes](#11-pendientes)

---

## 1. Novedades

Cambios de octubre de 2026 (desarrollados en `pintorV3_ws` y pasados a
`pintor_ws` el 2026-10-12). La versión anterior completa está respaldada en
`~/respaldos/pintor_ws_antes_v3_2026-10-12/`.

**Launch único** — `ur5_algoritmos/launch/pintor.launch.py`
- `modo:=sim | real` con todo lo necesario para cada caso (ver §3).
- `modelo_robot:=ur5 | ur5e`: un solo parámetro elige el robot de Gazebo o del
  driver **y** los parámetros DH que usan todos los nodos para la cinemática.
- Parámetros de dibujo expuestos (velocidad, offsets, pasada en el aire…) y
  ejemplos de uso comentados en la cabecera del archivo.

**Trayectorias con series de Fourier**
- `trayectoria:=fourier` — texto: igual que `letter`, pero cada trazo es una
  serie de Fourier en vez de un spline (serie coseno para trazos abiertos,
  compleja para cerrados).
- `trayectoria:=fourier_img` — **imágenes nuevas**: dibuja los bordes de un
  logo o dibujo animado (líneas, contornos de zonas negras y bordes de color).
- Esqueleto mejorado: centrado sub-píxel (error medio 35 → 11 µm) y extremos
  alargados hasta el final real del trazo, descontando el radio del plumón.

**Control de `move_draw_sub`**
- Nuevo seguimiento a **velocidad constante** (`velocidad_dibujo`, por
  defecto 20 mm/s), con término de avance e integración con el tiempo real.
  Si el robot se atrasa más de `error_max_avance`, la referencia espera, así
  los detalles pequeños no se recortan. El modo anterior sigue disponible
  (`modo_seguimiento:=puntos`).
- **Jacobiano analítico** (igual al numérico, 4 veces más rápido).
- Al terminar, el plumón **sube 30 mm en vertical** antes de volver al
  inicio. Antes volvía arrastrándose sobre la hoja y la rayaba en diagonal.
- Bajada al papel con 0.3 mm de tolerancia (antes 0.1 mm, era lo más lento).

**Simulación**
- Controladores compatibles con `move_draw_sub` sin modificarlo, una mesa
  con hoja A4 a la altura calibrada y una postura inicial sobre la hoja.
- **Trazo del plumón en RViz** (`/pintor/trazo`), también en modo real.
- Gazebo y RViz en la **GPU NVIDIA** (PRIME offload).
- Unas **7 veces más rápida**: `move_draw_sub` usa el reloj de la PC y no
  el de Gazebo, y NumPy corre en un solo hilo. Ejemplos medidos: "Sergio",
  342 s → 48 s; Bob Esponja, 3 min 38 s; logo UTEC, 3 min 26 s.

---

## 2. Requisitos y compilación

- Ubuntu 22.04, ROS 2 Humble, Gazebo Classic 11
- Paquetes ROS: `ur_robot_driver`, `ur_description`, `ur_controllers`,
  `gazebo_ros`, `gazebo_ros2_control`, `controller_manager`
- Python: `numpy`, `scipy`, `opencv-python`, `scikit-image`, `Pillow`,
  `osqp`, `PyYAML`, `matplotlib`

```bash
pip3 install --user numpy scipy opencv-python scikit-image Pillow osqp pyyaml matplotlib
```

> `scikit-image` se importa como `skimage`, pero **`pip install skimage`
> falla**: el paquete se llama `scikit-image`.

Compilar:

```bash
cd ~/pintor_ws
colcon build --symlink-install
source install/setup.bash
```

Con `--symlink-install`, los cambios en los `.py` y en los launch se aplican sin
recompilar. Hay que recompilar solo si se agrega un nodo nuevo a `setup.py`.

---

## 3. Cómo funciona el launch

### `modo:=sim`

```
Gazebo (UR5/UR5e + mesa con hoja A4) + RViz + controladores
   │   espera_sim (10 s)
   ▼
Lleva el robot a la postura inicial (30 mm sobre el centro de la hoja)
   ▼
move_draw_sub (Z plana = z_sim) + nodo de trayectoria + trazo en RViz
   ▼
Dibuja → levanta 30 mm → vuelve sobre el inicio
```

- **No calibra**: Gazebo no tiene sensor de fuerza. La Z de dibujo es la del
  centro de la última calibración guardada
  (`src/calibracion/surface_calibration_A4.npz`), o la que se pase con `z_sim`.
- El robot arranca sobre el centro de la hoja, y los nodos usan esa posición
  como centro del dibujo.

### `modo:=real`

```
ur_robot_driver (robot_ip) + RViz
   │   espera_driver (5 s) → dar PLAY a External Control en el teach pendant
   ▼
calibration_surface_A4_4x4: palpa 16 puntos (malla 4x4) con el sensor de fuerza
   │   centrada en la posición ACTUAL del robot; al final vuelve a ella
   │   guarda src/calibracion/surface_calibration_A4.{npz,csv,txt}
   ▼
¿El .npz se escribió en ESTA corrida?
   ├─ no  → ERROR, no se dibuja (la calibración falló o se interrumpió)
   └─ sí  → move_draw_sub (Z = superficie z(x,y)) + nodo de trayectoria + trazo en RViz
              ▼
            Dibuja → levanta 30 mm → vuelve sobre el inicio
```

- **Antes de lanzar, deja el plumón sobre el centro de la hoja, un poco
  levantado.** Ese punto es el centro de la calibración y del dibujo.
- Con `lanzar_driver:=false` se usa un driver que ya esté corriendo.

### Trayectorias (`trayectoria:=`)

| Valor | Nodo | Qué dibuja |
|---|---|---|
| `fourier_img` | `fourier_image_trajectory` | **Imagen** (`imagen:=ruta`): solo bordes, con Fourier |
| `fourier` | `fourier_trajectory` | **Texto** (`texto:=...`) con Fourier |
| `letter` | `letter_trajectory` | Texto con splines (versión anterior) |
| `painting` | `painting_trajectory` | Contornos por color de `~/Downloads/utec_test.jpg` (fijo) |
| `img` | `img_trajectory` | Imagen con el pipeline anterior |

Todos publican en `letter_trajectory/xy` y `letter_trajectory/flags`, y
`move_draw_sub` ejecuta lo que reciba.

---

## 4. Ejemplos de uso

```bash
source ~/pintor_ws/install/setup.bash

# Simulación: una palabra
ros2 launch ur5_algoritmos pintor.launch.py modo:=sim modelo_robot:=ur5e \
    trayectoria:=fourier texto:=Sergio letra_offset_y:=0.0

# Simulación: un dibujo (solo bordes) a 20 mm/s
ros2 launch ur5_algoritmos pintor.launch.py modo:=sim modelo_robot:=ur5e \
    trayectoria:=fourier_img imagen:=/home/mito/Downloads/UTEC-Logo.jpg \
    ancho_mm:=165 alto_mm:=230 ancho_plumon_mm:=0.8 velocidad_dibujo:=0.02

# Robot real: PASADA EN EL AIRE (10 mm sobre el papel, lento)
ros2 launch ur5_algoritmos pintor.launch.py modo:=real modelo_robot:=ur5e \
    robot_ip:=192.168.56.102 trayectoria:=fourier_img imagen:=/ruta/dibujo.jpg \
    velocidad_dibujo:=0.01 offset_tapa:=0.010 umbral_trazo:=0.012

# Robot real: dibujo de verdad (empezar con 0.01 m/s y luego subir)
ros2 launch ur5_algoritmos pintor.launch.py modo:=real modelo_robot:=ur5e \
    robot_ip:=192.168.56.102 trayectoria:=fourier_img imagen:=/ruta/dibujo.jpg \
    ancho_plumon_mm:=0.8 velocidad_dibujo:=0.01 offset_contacto:=0.0004 \
    z_min_seguridad:=0.17
```

---

## 5. Argumentos del launch

### General
| Argumento | Por defecto | Descripción |
|---|---|---|
| `modo` | `sim` | `sim` (Gazebo, sin calibrar) o `real` (driver + calibración 4x4) |
| `modelo_robot` | `ur5e` | `ur5` o `ur5e`: robot de Gazebo/driver **y** DH de la cinemática de todos los nodos |
| `trayectoria` | `painting` | `fourier_img`, `fourier`, `letter`, `painting`, `img` |

### Qué dibujar
| Argumento | Por defecto | Descripción |
|---|---|---|
| `imagen` | — | `fourier_img`: ruta de la imagen |
| `ancho_mm`, `alto_mm` | `160`, `160` | Caja máxima del dibujo en el papel [mm]. Se conserva la proporción. **Incluye los márgenes blancos de la imagen** |
| `ancho_plumon_mm` | `0.8` | Ancho **real** de la punta [mm]. Con él se ubican los contornos y se calcula cuánto alargar los extremos |
| `texto` | — | `fourier` / `letter`: palabra a escribir |
| `letra_offset_y` | — | Desplazamiento en Y [m] del texto. El nodo usa −0.15 por defecto; `0.0` lo centra |
| `n_armonicos` | `0` | `fourier`: armónicos por trazo; `0` = automático según el largo |

### Cómo dibujar (`move_draw_sub`)
| Argumento | Por defecto | Descripción |
|---|---|---|
| `velocidad_dibujo` | `0.02` | Velocidad del plumón sobre el papel [m/s]. **En el real, empezar con 0.01** |
| `modo_seguimiento` | `velocidad` | `velocidad` (constante) o `puntos` (control anterior) |
| `error_max_avance` | `0.0005` | Si el robot se atrasa más que esto [m], la referencia espera |
| `offset_contacto` | `0.0004` | Sube [m] la Z calibrada (la calibración mide con el plumón comprimido) |
| `offset_tapa` | `0.0` | Sube [m] todo el dibujo. `0.010` = pasada en el aire |
| `altura_regreso` | `0.03` | Al terminar, sube esto [m] en vertical antes de volver |
| `umbral_trazo` | — | Altura sobre el papel que el trazo de RViz cuenta como dibujo. Vacío = 1.5 mm en sim y 3 mm en real; para la pasada en el aire usar `0.012` |

### Solo sim
| Argumento | Por defecto | Descripción |
|---|---|---|
| `z_sim` | — | Z plana [m]; vacío = la del centro de la última calibración |
| `usar_gpu` | `true` | Gazebo y RViz en la NVIDIA |
| `gazebo_gui` | `true` | Ventana de Gazebo |
| `espera_sim` | `10.0` | Segundos de arranque de Gazebo antes de dibujar |
| `largo_plumon` | `0.0` | Solo para ubicar la hoja en Gazebo (distancia brida–punta) |

### Solo real
| Argumento | Por defecto | Descripción |
|---|---|---|
| `robot_ip` | `192.168.56.102` | IP del robot (la PC va en 192.168.56.100) |
| `lanzar_driver` | `true` | `false` = usar un driver ya abierto |
| `espera_driver` | `5.0` | Segundos antes de calibrar (tiempo para dar play a External Control) |
| `z_min_seguridad` | `0.10` | Calibración: Z mínima [m] a la que puede bajar si el sensor no detecta el contacto. Es absoluta (marco de la FK) y **depende de `modelo_robot`**: ponerla 1–2 cm bajo la Z del papel que reporta la calibración |

---

## 6. Robot real: consideraciones y puesta en marcha

En orden de cuánto pueden deformar el dibujo:

### 6.1 La cinemática debe ser la del robot
- `modelo_robot` tiene que coincidir con el robot. Si no, el dibujo sale
  desplazado y deformado. El robot del laboratorio es un **UR5e**.
- Los DH del `ur5e` son los **nominales de fábrica**, no los de calibración
  de este robot en particular. Si el dibujo sale algo corrido o deformado
  hacia los bordes, el siguiente paso es extraer los parámetros reales con
  `ur_calibration` y usarlos en la cinemática.
- La calibración 4x4 depende del modelo con que se midió. En modo real se
  recalibra en cada corrida, así que no es problema.

### 6.2 El lazo de control corre en la PC
- `move_draw_sub` es Python y manda posiciones por la red a unos 200 Hz. Si un
  ciclo se retrasa, el brazo da un tirón. Durante el dibujo conviene cerrar
  lo pesado (Gazebo, navegador). RViz no molesta.
- El modo velocidad mide el error contra la posición **que manda**, no contra
  la que el robot alcanza de verdad. A 10–20 mm/s la diferencia debería ser
  pequeña. Si en el real ves esquinas redondeadas, baja `velocidad_dibujo`.
- Deja el control de velocidad del teach pendant al **100 %**.

### 6.3 El plumón
- **Soporte con resorte**: absorbe 1–2 mm de error en Z. Con un plumón
  rígido, cualquier error se vuelve demasiada presión (arrastra y deforma) o
  muy poca (deja huecos).
- **Velocidad y tipo de plumón**: uno grueso aguanta 20 mm/s; uno fino o
  gastado deja huecos. Prueba en una hoja aparte.
- **`ancho_plumon_mm`** debe ser el ancho real de la punta.
- **`offset_contacto`**: si el plumón raspa o se aprieta, súbelo; si deja
  huecos, bájalo. Ajústalo con trazos de prueba.

### 6.4 La hoja
- El dibujo se centra donde está el robot al arrancar, porque la calibración
  vuelve a ese punto al terminar.
- Tiene que quedar dentro de la zona calibrada: la malla cubre unos
  170 × 257 mm, y fuera de ella la Z se extrapola. Como referencia, un
  dibujo de 150 × 125 mm entra con margen.

### 6.5 Checklist de puesta en marcha
1. **External Control** en el teach pendant debe apuntar a la IP de la PC
   (**192.168.56.100**). Según lo anotado, hoy apunta a 192.168.10.103.
2. Plumón sobre el centro de la hoja, un poco levantado.
3. Mano en la **parada de emergencia** durante la calibración. Si el sensor
   de fuerza no detecta el contacto, la bajada solo se detiene en
   `z_min_seguridad` (por defecto 0.10 m absolutos). Con el papel a ~0.19 m,
   eso son unos 9 cm de margen. Conviene pasarlo en el launch, 1–2 cm bajo la
   Z del papel (`z_min_seguridad:=0.17`). La Z del papel aparece en
   `src/calibracion/surface_calibration_A4.txt`. Ojo: cambia con
   `modelo_robot`, así que hay que tomarla de una calibración hecha con el
   mismo modelo.
4. **Pasada en el aire**: `offset_tapa:=0.010 umbral_trazo:=0.012
   velocidad_dibujo:=0.01`. Revisa el recorrido en RViz sin tocar la hoja.
5. Primer dibujo real con `velocidad_dibujo:=0.01`; si sale bien, sube a 0.02.
6. El trazo de RViz usa la posición **real** del robot (TF), así que sirve
   para comparar lo dibujado con lo planeado.

---

## 7. Notas de la simulación

- **Controladores**: `config/pintor_sim_controllers.yaml` declara un
  `JointTrajectoryController` normal con el nombre
  `scaled_joint_trajectory_controller`, porque Gazebo no tiene la interfaz
  `speed_scaling` del controlador original de UR. También declara
  `forward_position_controller` inactivo. Así `move_draw_sub` funciona igual
  que con el robot real.
- **Postura inicial**: `config/initial_positions_pintor_<modelo>.yaml`. El
  brazo cede por gravedad entre que aparece en Gazebo y que se activan los
  controladores (9 a 30 mm). Por eso el launch lo vuelve a llevar a esa
  postura antes de dibujar.
- **Mesa y hoja**: se crean a la altura `z_sim`, en el centro de la última
  calibración.
- **GPU**: con `usar_gpu:=true` se fijan `__NV_PRIME_RENDER_OFFLOAD=1` y
  `__GLX_VENDOR_LIBRARY_NAME=nvidia`. Se comprueba con `nvidia-smi`:
  `gzserver`, `gzclient` y `rviz2` deben aparecer en la lista.
- **Modelos del world**: el launch agrega `models/` a `GAZEBO_MODEL_PATH`. Si
  no, Gazebo busca los modelos en internet y no termina de arrancar.

---

## 8. Generación de trayectorias (Fourier)

### Texto (`fourier`)
texto → imagen A4 → esqueleto → **refinado** (centrado sub-píxel + extremos)
→ grafo → orden y fusión → **serie de Fourier por trazo**.
- Trazo abierto: serie coseno (DCT), que no oscila en los extremos.
- Trazo cerrado: serie compleja (epiciclos).
- Armónicos: automático, 6 por cm de trazo.

Frente a los splines, la precisión es la misma (0.13 frente a 0.16 mm) y la
última etapa es 10 veces más rápida. El tiempo total casi no cambia, porque
lo que más tarda es renderizar el texto.

### Imagen (`fourier_img`, solo bordes)
1. **Líneas finas** (más oscuras que su entorno; black-hat de 2 mm): se
   dibujan por su línea central (esqueleto refinado).
2. **Zonas negras** más gruesas que 2 mm (pupilas, zapatos, letras grandes):
   solo su **contorno**, a medio plumón hacia adentro del borde.
3. **Bordes de color** sin línea encima (manchas, zonas celestes): fronteras
   de k-means, alargadas hasta tocar el contorno negro.
4. Orden por vecino más cercano. Los trazos solo se fusionan si el conector
   cae sobre tinta.
5. Fourier por trazo (10 armónicos/cm), con un waypoint cada 0.6 mm.

Resolución de trabajo: 14 px/mm, suficiente para que se lea texto de ~3–4 mm.
Los parámetros finos están en `IMAGE_DEFAULTS`
(`trayectorias/fourier_image_functions.py`). Dos que no expone el launch:
- `rellenar_negros`: `true` rellena las zonas negras.
- `sombreado`: `true` raya en zigzag las zonas oscuras.

---

## 9. Archivos del proyecto

```text
src/
├── README.md                       ← este archivo
├── calibracion/                    ← datos de calibración y logs
│   ├── surface_calibration_A4.npz  ← superficie z(x,y) (la usa el dibujo y el modo sim)
│   ├── surface_calibration_A4.{csv,txt}
│   └── posiciones_robot.txt        ← log de posiciones del último dibujo
├── letras_png_A4/                  ← imágenes que genera letter/fourier
├── ur5_algoritmos/
│   ├── launch/
│   │   ├── pintor.launch.py        ← ★ launch único (sim | real)
│   │   ├── pintor_completo.launch.py, dibujo_superficie.launch.py, ...  (anteriores)
│   └── ur5_algoritmos/
│       ├── move_draw_sub.py        ← ejecuta el dibujo (modo velocidad, regreso seguro)
│       ├── calibration_surface_A4_4x4.py
│       ├── fourier_trajectory.py           ← nodo texto con Fourier
│       ├── fourier_image_trajectory.py     ← nodo imagen con Fourier
│       ├── letter_trajectory.py, painting_trajectory.py, img_trajectory.py
│       ├── cinematica/
│       │   ├── fk_functions.py     ← DH de ur5 y ur5e (PINTOR_UR_MODELO)
│       │   ├── jacobiano_analitico.py
│       │   └── ik_functions.py, kine_control_functions.py, QP_functions.py
│       └── trayectorias/
│           ├── fourier_trajectory_functions.py  ← series + refinado del esqueleto
│           ├── fourier_image_functions.py       ← pipeline de imagen
│           └── letter_trajectory_functions.py, ...
└── ur_simulation_gazebo/
    ├── launch/pintor_sim.launch.py ← Gazebo + RViz + controladores + mesa
    ├── config/
    │   ├── pintor_sim_controllers.yaml
    │   ├── initial_positions_pintor_ur5.yaml / _ur5e.yaml
    │   └── pintor.rviz             ← vista del UR + trazo
    ├── scripts/trazo_rviz.py       ← publica /pintor/trazo
    ├── world/, models/
```

---

## 10. Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| Gazebo no termina de arrancar, `/spawn_entity unavailable` | Faltan los modelos del world en `GAZEBO_MODEL_PATH`. `pintor_sim.launch.py` lo agrega; si usas otro launch, agrégalo a mano |
| En sim el dibujo va lentísimo | `move_draw_sub` está usando el reloj de Gazebo. `pintor.launch.py` lo lanza con el reloj de la PC; revisa que lo estés usando |
| El trazo de RViz muestra rayas que no son del dibujo | Son viajes del robot a menos de `umbral_trazo` del papel. Revisa que el robot arranque por encima de la hoja |
| La FK no coincide con el robot (dibujo desplazado) | `modelo_robot` equivocado (ver §6.1) |
| `ERROR: ... es de una calibración anterior` | La calibración de esta corrida falló o se interrumpió, y se rehúsa a dibujar con datos viejos |
| `UserWarning: Unable to import Axes3D` | Inofensivo: hay dos matplotlib instalados (apt y pip) |
| `AMENT_PREFIX_PATH ... doesn't exist` | Quedó una ruta vieja en la terminal. Abre otra terminal y vuelve a hacer `source` |

---

## 11. Pendientes

- Parámetros DH reales del UR5e (`ur_calibration`) en vez de los nominales.
- Cerrar el lazo del modo velocidad con `/joint_states`, para que el error
  se mida contra la posición real del robot.
- Definir el `z_min_seguridad` correcto para el UR5e, tomando la Z del papel
  de una calibración hecha con `modelo_robot:=ur5e`, y revisar el signo del
  umbral de fuerza.
- Probar en el robot real todo lo nuevo; hasta ahora solo está validado en
  simulación.
