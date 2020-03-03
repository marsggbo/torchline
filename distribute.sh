rm -rf build dist torchline.egg-info
python setup.py sdist bdist_wheel
if (($1==1));then
    python -m twine upload dist/*
elif (($1==2));then
    python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
else
    echo "Wrong command, only support 1 or 2"
fi