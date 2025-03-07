from setuptools import setup, find_packages

setup(
    name='shamans',
    version='0.1.0',
    author='Diego DI CARLO',
    author_email='diego.dicarlo89@gmail.com',
    description='Sound Source Localization with NSteerer and AlphaStable',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'einops',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'export-models=shamans.main:main',  # Assuming you have a main function in export_models.py
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)