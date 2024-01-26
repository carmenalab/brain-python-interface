import setuptools

setuptools.setup(
    name="aolab-bmi3d",
    version="1.0.1",
    author="Lots of people",
    description="electrophysiology experimental rig library",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "django==4.1",
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
