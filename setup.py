from setuptools import setup, find_packages


setup(
        name="multi-object-fetch",
        version="0.1",
        packages=find_packages(),
        install_requires=[
            "cython<3",
            "gym<=0.17.3",
            "mujoco-py", 
            "opencv-python",
            "pynput",
            "tabulate",
            "tqdm",
            "numpy==1.24.3"
        ],
    )
