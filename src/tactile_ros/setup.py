from setuptools import find_packages, setup

package_name = 'tactile_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xinyun',
    maintainer_email='xinyun.chi22@gmail.com',
    description='GelSight tactile sensor node (FEATS force estimation) feeding the realrobot-plug-smolvla checkpoint rollout.',
    license='TODO',
    entry_points={
        'console_scripts': [
            'gelsight_node = tactile_ros.gelsight_node:main',
        ],
    },
)
