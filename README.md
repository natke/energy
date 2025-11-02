# Use CodeCarbon to measure AI on device

## Setup

```
pip install codecarbon
```

## Code

Import the following modules into Python code

```python
from codecarbon import track_emissions
```

Decorate code with `@track_emissions()`

```
@track_emissions()
def function():
```

## Run 

```
python model-chat.py -m <model> --verbose 2> error.log
```

## Notes

### Intel Mac
On Intel mac install InteL Power Gadget (deprecated)

This error occurs
```
ERROR: EnergyDriver_executeCommands [via readSample] returned 0xe00002bc
```

### Apple silicon Mac

On Apple silicon mac use Power Metrics (pre-installed)

If Power Gadget is not installed, Code Carbon will use Power Metrics but it does  not produce the right output format on Intel mac

```bash
sudo powermetrics -n 1 --samplers cpu_power --format csv -i 1 -o test.log
```

