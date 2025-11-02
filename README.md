# Use CodeCarbon to measure AI on device

## Setup

```
pip install codecarbon
```

## Code

Import the following modules into Python code

```python
from typing import Optional, Sequence
from codecarbon import track_emissions
```

Decorate code with `@track_emissions()`

```
@track_emissions()
def function():
```

## Run 

```
python model-chat.py -m <model>
```

## Notes

Power metrics not producing the right output format on Intel mac

```bash
sudo powermetrics -n 1 --samplers cpu_power --format csv -i 1 -o test.log
```