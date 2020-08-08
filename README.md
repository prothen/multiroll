# railway_rollout
Railway rollout for flatland Neurips Challenge 2020


## Usage
Execute with
```
python testbed.py --display True
```
where the `--display` flag allows to disable and enable the visualisation via GUI.

And profile the code performance using
```
cd profiler && ./generate_pstats_profile.sh
```
which will show a graphical representation of the most time consuming methods.

## The environment
Install with `pip install flatland-rl` or clone
```
git clone http://gitlab.aicrowd.com/flatland/flatland.git
```
and install with `python setup.py install` in the root of the repository.


## References
- [Flatland information metrics](https://flatland.aicrowd.com/faq/env.html)
