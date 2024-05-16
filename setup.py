from setuptools import find_packages,setup

setup(
    name="Machine Learning Model's Parameter Tuning",
    version='0.0.0',
    author='KRSNA',
    author_email='krisnabadde@gmail.com',
    install_requires=['scikit-learn==1.4.2','matplotlib==3.9.0','streamlit==1.34.0'],
    packages=find_packages()
)