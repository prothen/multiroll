# railway_rollout
Railway rollout for flatland Neurips Challenge 2020


## Usage
Execute with
```
python testbed.py --display True --step True
```
- the flag `--display` allows to disable and enable the visualisation via GUI,
- the flag `--skip` allows to step through each environment step.

And profile the code performance using
```
cd profiler && ./generate_pstats_profile.sh
```
which will show a graphical representation of the most time consuming methods.

## Installation
Install the dependencies with
```
pip install --requirements.txt
```

## Contribution
**Author**: Philipp Rothenhäusler (philipp.rothenhaeusler a t gmail.com)

**Maintainer**: Philipp Rothenhäusler (philipp.rothenhaeusler a t gmail.com)

## LICENSE
All rights are reserved.
