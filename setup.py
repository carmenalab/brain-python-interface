import setuptools

setuptools.setup(
    name="bmi3d",
    version="0.3.0",
    author="Lots of people",
    description="electrophysiology experimental rig library",
    packages=setuptools.find_packages(),
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
        "pygame",
        "nitime",
        "tornado",
        "tables",
        "h5py",
        "pymysql",
    ]
)
