### Instructions for building using pyproject.toml and meson

Fairly rough notes so far, 

# Setup

`pip install build`


# Building 

`python -m build` (will build both sdist and wheel)

# Specifying architecture on mac

``` bash
export ARCHFLAGS="-arch x86_64"
python -m build
```
alternatively / preferentially `-arch aarch64`



# Validating that everything made it into the wheel

- Unpack wheel
- e.g.
``` bash
diff -rq --exclude='*.pyc' --exclude='__pycache__' --exclude='*.so' --exclude='*.build' --exclude='.DS_Store' --exclude='*.h' --exclude='*.c' ./PYME ./dist/python_microscopy-25.6.5-cp313-cp313-macosx_10_16_x86_64/PYME
```

It's a bit of a judgement call what should make it in (I exclude a bunch of the bitrot and super-experimental stuff by default), but if you've added something new and it doesn't make it through this should pick up on it.
