# multiroll
Reinforcement learning approach to Flatland Neurips 2020 challenge using multi-agent rollout with graph based NetworkX abstraction.

_Note: Repository is in a development stage but may be continued, if you encounter issues please let me know by opening an issue. It is very likely to be minor code inconsistencies or small obstacles from the setup process._


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
