from setuptools import setup

install_requires = [
    'torch',
    'numpy'
    ]

# Get version from the module
with open('pytorch_tcn/__init__.py') as f:
    for line in f:
        if line.find('__version__') >= 0:
            version = line.split('=')[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

setup(
    name='pytorch-tcn',
    version=version,
    description='Pytorch TCN',
    author='Paul Krug',
    url='https://github.com/paul-krug/pytorch-tcn',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['pytorch_tcn'],
    install_requires=install_requires
)