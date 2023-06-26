import setuptools

setuptools.setup(
    name="aolab-bmi3d",
    version="0.9.6",
    author="Lots of people",
    description="electrophysiology experimental rig library",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "django",
        "celery",
        "jinja2",
        "scipy",
        "traits",
        "pandas",
        "patsy",
        "statsmodels",
        "pygame",
        "PyOpenGL",
        "pylibftdi",
        "sphinx",
        "numpydoc",
        "tornado",
        "tables",
        "h5py",
        "pymysql",
    ]
)
