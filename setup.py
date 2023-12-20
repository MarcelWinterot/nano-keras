from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='nano-keras',
    version='1.2.0',
    packages=find_packages(),
    url='https://github.com/MarcelWinterot/nano-keras',
    license='MIT',
    author='Marcel Winterot',
    author_email='m.winterot1@gmail.com',
    description='Deep learning library made with numpy in the style of Keras API',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
    ],
    install_requires=['numpy'],
    keywords=['python', 'machine-learning',
              'machine-learning-library', 'keras', 'numpy']
)
