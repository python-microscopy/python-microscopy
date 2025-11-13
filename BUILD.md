### Instructions for building using pyproject.toml and meson

Fairly rough notes so far, 

# Setup

`pip install build`


# Building 

`python -m build` (will build both sdist and wheel)

In most cases you might be better building just the wheel, ie

`python -m build --wheel`

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

Obviously substituting the path to the un-packed wheel. It's a bit of a judgement call what should make it in (I exclude a bunch of the bitrot and super-experimental stuff by default), but if you've added something new and it doesn't make it through this should pick up on it.


## Editable install (replaces `python setup.py develop`)

Ensure `build`,  `meson` and `meson-python` are installed alongside standard install dependencies

```bash
pip install --no-build-isolation --editable .  
```

To recompile extension modules:

```bash
meson compile -C build/cp313
```

replacing the build subdir (e.g. `cp313`) with the one appropriate for your python version (i.e. the one that was generated with `pip install -e` above)