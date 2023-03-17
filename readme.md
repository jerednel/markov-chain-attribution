 
## Markov Chains for Attribution Modeling

This is a proof-of-concept I built out that leverages a first order Markov chain to reallocate conversions in the manner explained by [Anderl, Eva and Becker, Ingo and Wangenheim, Florian V. and Schumann, Jan Hendrik](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2343077) in " Mapping the Customer Journey: A Graph-Based Framework for Online Attribution Modeling"

If this concept is new, check out the post on [markov chain attribution modeling](https://blog.nel.sh/building-an-attribution-model-with-markov-chains/).   

Big thanks to people writing and commenting on this [AnalyzeCore article](https://analyzecore.com/2016/08/03/attribution-model-r-part-1/) for a very useful article and comment section.

### Motivation

There is an amazing R package called ChannelAttribution which does this as well as higher-order models very well.  My day-to-day workflow is centered around Python so I wanted to build out a version to A) have something I can go-to for connecting directly to SQL tables to path journeys and B) better understand the process by which these attribution models are generated.  ChannelAttribution makes it very easy to faceroll a fractional attribution model without really understanding what's going on.  Which is great!  But if you want to understand better, building your own tends to help.

To get started quickly you can install via pip.

### Installation
Make sure your version is at least 0.4 - prior versions included suboptimal ways of generating our initial transition states that resulted in negative conversions in large datasets.
```#python
pip install markov-model-attribution
```

### Use
* This package currently accepts a single-column Pandas dataframe. 
* Each path should begin with "start" and end with either "conv" or "null".
* Each path should be delimited by " > "

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
