import setuptools

setuptools.setup(
    name="bmi3d",
    version="0.1.0",
    author="people",
    description="electrophysiology experimental rig library",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "numexpr",
        "cython",
        "django-celery",
        "traits",
        "pandas",
        "patsy",
        "statsmodels",
        "PyOpenGL",
        "Django",
        "pylibftdi",
        "nitime",
        "sphinx",
        "numpydoc",
        "tornado",
        "tables",
        "pyserial",
        "h5py",
        "ipdb",
    ]
)
