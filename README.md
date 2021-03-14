# multiroll
The multi-agent rollout Python implementation.


## Usage
Execute with
```
python testbed.py
```
- use `--display False` to disable the visualisation,
- use `--skip  False` to disable user prompts.
- use `--roll  True` to enable the multi-agent rollout.

## Installation
Install the dependencies with
```
pip install --requirements.txt
```

And install the multiroll package with
```
pip install .
```

_Note: For development installations it is recommended to instead run `./init.sh` in order to add the package path to the `PYTHONPATH` and avoid updating the installation folder on source code changes._

## Profiling
And profile the code performance using
```
./profile.sh
```
which will show a graphical representation of the most time consuming methods.


## Contribution
Any contribution is welcome.
If you find missing instructions or something did not work as expected please create an issue and let me know.

## License
See the `LICENSE` file for details of the available open source licensing.
