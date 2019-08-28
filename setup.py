from setuptools import find_packages, setup

setup(name='deeplearning',
      version='0.1',
      description='Utils for use in deep learning',
      url='https://github.com/amitkml/deeplearning',
      author='Amit Kayal',
      author_email='amitkayal@outlook.com',
      license='MIT',
      install_requires=[
          'numpy','pandas','beautifulsoup4','fastnumbers','more-itertools',
            'dill','stockstats','pytidylib','seaborn','gensim','nltk','fastnumbers',
            'joblib','Pygments','opencv-python',
      ],
      keywords=['Pandas','numpy','data-science','IPython', 'Jupyter','ML','Machine Learning','Deep Learning','Computer Vision','Keras'],
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
