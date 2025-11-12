# Tierney_Python_Portfolio
This is my final project for the first python series 

## Using Jupyter Notebooks 1

In this analysis, we downloaded data to be used further in the class, and sorted numbers and created grapths

```python
%matplotlib inline
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sms 
sms.set(style = "darkgrid")
```


```python
df = pd.read_csv("/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv")
```


```python
df.head()
```


```python
df.tail()
```


```python
df.columns = ["year", "rank", "company", "revenue", "profit"]
```


```python
df.head()
```


```python
len(df)
```


```python
df.dtypes
```


```python
non_numeric_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numeric_profits].head()
```


```python
set(df.profit[non_numeric_profits])
```


```python
len(df.profit[non_numeric_profits])
```


```python
bin_sizes, _,_= plt.hist(df.year[non_numeric_profits], bins= range(1955, 2006))
```


```python
df = df.loc[-non_numeric_profits]
df.profit = df.profit.apply(pd.to_numeric)
```


```python
len(df)
```


```python
df.dtypes
```


```python
group_by_year = df.loc[:, ("year", "revenue", "profit")].groupby("year")
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y) 
    ax.margins(x = 0, y = 0)
```


```python
fig, ax = plt.subplots()
plot (x, y1, ax, 'increase in mean fortune 500 company profits from 1955 to 2005', 'profit(millions)')
```


```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'increase in mean fortune 500 company revenues from 1955 to 2005', 'Revenue (millions)') 
```


```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha = 0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots (ncols= 2)
title = 'Increase in mean and std Fortune 500 company profits from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title, 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title, 'Revenue (millions)')
fig.set_size_inches(14,4)
fig.tight_layout()
```

