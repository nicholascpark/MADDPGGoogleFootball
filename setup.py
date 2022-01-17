from distutils.core import setup

setup(name='rldm',
      version='1.0',
      description='Reinforcement Learning and Decision Making',
      author='Miguel Morales',
      author_email='mimoralea@gatech.edu',
      url='https://github.gatech.edu/rldm/p3_advanced',
      packages=['rldm'],
      install_requires=['ray[default]==1.6', 'ray[rllib]==1.6',], # requirements are installed on Docker container
)
