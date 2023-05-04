Introducing PlotGPT, a lightweight AI assistant to help you visualize your data.

![](demo.gif)

Built on top of langchain and openai


Currently it will (most likely) only support plotting in matplotlib and seaborn

# Quickstart

To get started, all you need to do is give PlotGPT a pandas dataframe to inspect.

```python
import seaborn as sns
df = sns.load_dataset('iris')

ai = PlotGPT()
ai.inspect(df)
```

And then prompt it to start plotting.

```python
ai.ask("plot sepal width vs sepal length")
```


PlotGPT will remember previous prompts (up to a certain point), so you can build off of previous plots, mimicking iterative explorate data analysis.

```python
ai.ask("now colored it by specied")
```


By default, PlotGPT will return the plotting code. You can turn it off by setting the `show_code` flag.
```python
ai = PlotGPT(show_code=False)
```
> However, it is this author's recommendation to leave this setting on be able to sanity check the plots.


# Questions

## Should this completely replace my data analysis?

Probably not. Since this is an LLM, it's definitely not perfect. I'm more envisioning this as empowering initial discovery, where we use the AI to create a series of initial exploratory visualizations, and then have the human take over once you want to go deeper / more nuanced.
