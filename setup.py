import os
import io
import setuptools


here = os.path.dirname(__file__)

with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()


def install_deps():
    """
    Reads requirements.txt and preprocess it to be feed into setuptools.

    Returns:
         list of packages and dependency links.
    """
    default = open('requirements.txt', 'r').readlines()
    new_pkgs = []
    links = []
    for resource in default:
        if 'git+ssh' in resource:
            pkg = resource.split('#')[-1]
            links.append(resource.strip() + '-9876543210')
            new_pkgs.append(pkg.replace('egg=', '').rstrip())
        else:
            new_pkgs.append(resource.strip())
    return new_pkgs, links


pkgs, new_links = install_deps()

setuptools.setup(
    name='predictme',
    description='ML-based object prediction library',
    version='0.1.0',
    long_description=long_description,
    url='',
    license='MIT',
    author='Vitalii Kulanov, 2020',
    author_email='vitaliy@kulanov.org.ua',
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=pkgs,
    dependency_links=new_links,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development :: Libraries',
    ],
)
