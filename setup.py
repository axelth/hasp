from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x and not x.startswith('./')]

setup(name='hasp',
      version="1.0",
      description="Project Description",
      packages=find_packages(),
      install_requires=requirements,
      extras_require={
          'dev': [
              'black',
              'coverage',
              'flake8',
              'pytest',
              'yapf',
              'mlflow',
          ]
      },
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      #scripts=['scripts/hasp-run'],
      zip_safe=False)
