import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'cafetero'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*'))
    ],
    install_requires=['setuptools'],
    py_modules=['functions'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='daniel@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
        entry_points={
        'console_scripts': [
        	'control_QR = cafetero.control_QR:main',
        	'command_gazebo = cafetero.command_gazebo:main',
        	'paneo_ur5_uniforme = cafetero.paneo_ur5_uniforme:main',
        	'show_qr = cafetero.show_qr:main',
        ],
    },
)
