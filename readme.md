## Markov Chains for Attribution Modeling

This is a proof-of-concept I built out that leverages a first order Markov chain to reallocate conversions in the manner explained by [Anderl, Eva and Becker, Ingo and Wangenheim, Florian V. and Schumann, Jan Hendrik](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2343077) in " Mapping the Customer Journey: A Graph-Based Framework for Online Attribution Modeling"

If this concept is new, check out the post on [markov chain attribution modeling](https://www.jnel.me/how-to-leverage-markov-chains-for-attribution/).   

In this package we run simulations of user journeys based on the values of the transition matrix to derive the removal effect conversion rate.  The higher the number of simulations the longer it takes since it needs to be run for each channel or tactic that you pass to the model.

### Motivation

There is an amazing R package called ChannelAttribution which does this as well as higher-order models very well.  My day-to-day workflow is centered around Python so I wanted to build out a version to A) have something I can go-to for connecting directly to SQL tables to path journeys and B) better understand the process by which these attribution models are generated.  ChannelAttribution makes it very easy to faceroll a fractional attribution model without really understanding what's going on.  Which is great!  But if you want to understand better, building your own tends to help.

To get started quickly you can install via pip.

### Installation
```#python
pip install markov-model-attribution
```

### Use
* This package currently accepts a single-column Pandas dataframe. 
* Each path should begin with "start" and end with either "conv" or "null".
* Each path should be delimited by " > "

The arguments to pass are ```niter``` and ```paths```, where the former is an integer representing the number of simulations to run and paths is the Pandas dataframe containing your paths.


```#python
import markov_model_attribution as mma
import pandas as pd

# generate a sample dataset
df = pd.DataFrame({'Paths':['start > rem > rem > conv',
                           'start > pro > sem > conv',
                           'start > pro > null',
                           'start > sem > conv',
                           'start > pro > pro > sem > rem > conv',
                           'start > pro > pro > null'
                           'start > aff > rem > conv',
                           'start > pro > pro > null',
                           'start > sem > sem > null']})

model = mma.run_model(niter=500, paths=df)

```

Once you have the model constructed you can access a couple of things to compare how a fractional model does against a standard last touch model.

You can access these via

```python

print(model['markov_conversions'])

# This outputs a dictionary containing the markov conversion count
# {'pro': 1, 'rem': 1, 'sem': 2}


print(model['last_touch_conversions'])
# This outputs the last touch conversions for comparison
# {'pro': 0, 'rem': 2, 'sem': 2}
```

You can also access the transition matrix that of the state changes for your data.

The columns containing "minus" represent the removal affect transition matrix.

```python
print(model['transition_matrix'])

#        paths      prob  minus_pro  minus_rem  minus_sem
# 0   pro>null  0.333333   0.000000   0.333333   0.333333
# 1    pro>pro  0.333333   0.000000   0.333333   0.333333
# 2    pro>sem  0.333333   0.000000   0.333333   0.333333
# 3   rem>conv  0.666667   0.666667   0.000000   0.666667
# 4    rem>rem  0.333333   0.333333   0.000000   0.333333
# 5   sem>conv  0.666667   0.666667   0.666667   0.000000
# 6    sem>rem  0.333333   0.333333   0.333333   0.000000
# 7  start>pro  0.666667   0.666667   0.666667   0.666667
# 8  start>rem  0.166667   0.166667   0.166667   0.166667
# 9  start>sem  0.166667   0.166667   0.166667   0.166667
```