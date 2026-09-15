from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ur5_algoritmos'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

    # La tipografía viaja con el paquete: los nodos la buscan junto al módulo.
    # Las familias se reparten en varios niveles (Playwrite_CU/static/*.ttf,
    # UTEC/Stag/Stag Sans/*.otf), así que se cubre cada profundidad y tanto
    # .ttf como .otf: el pipeline rasteriza con FreeType y acepta los dos.
    package_data={
        package_name: [
            f'Tipografia/{"*/" * depth}*{ext}'
            for depth in range(5)
            for ext in ('.ttf', '.otf', '.txt')
        ],
    },

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),

    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='utec',
    maintainer_email='molortegui@utec.edu.pe',
    description='UR5 control with custom algorithms',
    license='Apache License 2.0',

    extras_require={
        'test': [
            'pytest',
        ],
    },

    entry_points={
        'console_scripts': [
            'calibration_draw = ur5_algoritmos.calibration_draw:main',
            'calibration_surface_A4_3x3 = ur5_algoritmos.calibration_surface_A4_3x3:main',
            'move_draw_sub = ur5_algoritmos.move_draw_sub:main',
            'imagen_trajectory = ur5_algoritmos.imagen_trajectory:main',
            'letter_trajectory = ur5_algoritmos.letter_trajectory:main',
        ],
    },
)
