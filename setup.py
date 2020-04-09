from setuptools import setup, find_packages


setup(
    name='energypy',
    version='0.4.0',
    description='reinforcement learning for energy systems',
    author='Adam Green',
    author_email='adam.green@adgefficiency.com',
    url='http://www.adgefficiency.com/',
    packages=find_packages(exclude=['tests', 'tests.*']),
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'hypothesis'],
    install_requires=['Click'],
    entry_points='''
            [console_scripts]
            energypy-experiment=energypy.cli:cli
        '''
)
