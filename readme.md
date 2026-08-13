 
## Markov Chains for Attribution Modeling

This is a proof-of-concept I built out that leverages a first order Markov chain to reallocate conversions in the manner explained by [Anderl, Eva and Becker, Ingo and Wangenheim, Florian V. and Schumann, Jan Hendrik](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2343077) in " Mapping the Customer Journey: A Graph-Based Framework for Online Attribution Modeling"

If this concept is new, check out the post on [markov chain attribution modeling](https://jeremy.bearblog.dev/building-an-attribution-model-with-markov-chains/).   

Big thanks to people writing and commenting on this [AnalyzeCore article](https://analyzecore.com/2016/08/03/attribution-model-r-part-1/) for a very useful article and comment section.

### Motivation

There is an amazing R package called ChannelAttribution which does this as well as higher-order models very well.  My day-to-day workflow is centered around Python so I wanted to build out a version to A) have something I can go-to for connecting directly to SQL tables to path journeys and B) better understand the process by which these attribution models are generated.  ChannelAttribution makes it very easy to faceroll a fractional attribution model without really understanding what's going on.  Which is great!  But if you want to understand better, building your own tends to help.

To get started quickly you can install via pip.

### Installation
```#python
pip install markov-model-attribution
```

Tested against pandas 1.5, 2.0, 2.2 and 3.0.

> **Upgrading from 0.4 or earlier?** Those versions relied on pandas chained
> assignment (`df.loc[x][y] = val`), which stopped writing through in pandas 2.0.
> On any pandas newer than 1.5 they returned an all-zero transition matrix and
> `nan` for every conversion and removal effect, silently. If you are on pandas
> 2.0+, upgrade. Two behaviour changes come with the fix — see
> [Breaking changes](#breaking-changes).

### Use
* This package accepts a single-column Pandas dataframe (or a Series).
* Each path should begin with "start" and end with either "conv" or "null".
* Each path should be delimited by " > "
* Channel names may contain anything except the ">" delimiter.
* Your dataframe is not modified.

Paths that violate these rules raise a `ValueError` naming the offending row,
rather than failing later inside the linear algebra.

The argument to pass is ```paths```, where paths is the Pandas dataframe containing your paths.


```#python
import markov_model_attribution as mma
import pandas as pd

# generate a sample dataset
df = pd.DataFrame({'Paths':['start > cone > ctwo > cthree > conv',
                           'start > cone > null',
                           'start > ctwo > cthree > null']})

model = mma.run_model(paths=df)

```

Once you have the model constructed you can access a couple of things to compare how a fractional model does against a standard last touch model.

You can access these via

```python

print(model['markov_conversions'])

# This outputs a dictionary containing the markov conversion count
# {'cone': 0.2, 'cthree': 0.4, 'ctwo': 0.4}


print(model['last_touch_conversions'])
# This outputs the last touch conversions for comparison
# {'cone': 0, 'cthree': 1, 'ctwo': 0}
```

You can also access the removal effect matrix of the underlying result.  

```python
print(model['removal_effects'])

# {'cone': 0.5, 'cthree': 1.0, 'ctwo': 1.0}
```

The full transition matrix and absorption probabilities are also available via
`model['transition_matrix']` and `model['absorption_matrix']`.

### Breaking changes

**Channel names are no longer stripped.** Earlier versions ran every path
through `re.sub('[^a-zA-Z> ]', '', path)`, which silently deleted digits,
underscores and hyphens. `c1` and `c2` both became `c` and were merged into a
single channel; `email_promo_2024` became `emailpromo`. Names are now used
verbatim, so results will change for any dataset whose channel names contain
characters outside `a-z`. This is the intended behaviour, but it is a change —
compare against a previous run before you trust the delta.

**Input is now validated.** Paths that are missing `start`, missing a terminal
`conv`/`null`, or that place a reserved state mid-path now raise instead of
producing an `IndexError` or a quietly wrong number. Multi-column dataframes
raise rather than double-counting.

### Development

```#python
pip install pytest
python -m pytest tests/
```
