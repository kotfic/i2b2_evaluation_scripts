from setuptools import setup, find_packages
setup(    
    name="i2b2eval",
    author="Christoper Kotfila",
    author_email="ckotfila@albany.edu",
    license="APACHE",
    description="A package for evaluating Standoff Annotations in the i2b2 2014 shared task",
    version="1.2.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            'i2b2eval = i2b2eval.evaluate:main'
        ]
    }
)
