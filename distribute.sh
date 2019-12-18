rm -rf build dist torchline.egg-info
python setup.py sdist bdist_wheel
python -m twine upload dist/*
#python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*