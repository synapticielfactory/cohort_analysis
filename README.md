# Cohort Analysis with Python

> This article was inspired from [Greg Reda](http://www.gregreda.com/2015/08/23/cohort-analysis-with-python/), and the objectif is to adapt the content to the context of Elasticsearch.


Despite having done it countless times, I regularly forget how to build a cohort analysis with Python and pandas. I’ve decided it’s a good idea to finally write it out - step by step - so I can refer back to this post later on. Hopefully others find it useful as well.

I’ll start by walking through what cohort analysis is and why it’s commonly used in startups and other growth businesses. Then, we’ll create one from a standard purchase dataset.


## What is cohort analysis ?

A cohort is a group of users who share something in common, be it their sign-up date, first purchase month, birth date, acquisition channel, etc. Cohort analysis is the method by which these groups are tracked over time, helping you spot trends, understand repeat behaviors (purchases, engagement, amount spent, etc.), and monitor your customer and revenue retention.

It’s common for cohorts to be created based on a customer’s first usage of the platform, where "usage" is dependent on your business’ key metrics. For Uber or Lyft, usage would be booking a trip through one of their apps. For GrubHub, it’s ordering some food. For AirBnB, it’s booking a stay.

With these companies, a purchase is at their core, be it taking a trip or ordering dinner — their revenues are tied to their users’ purchase behavior.

In others, a purchase is not central to the business model and the business is more interested in "engagement" with the platform. Facebook and Twitter are examples of this - are you visiting their sites every day? Are you performing some action on them - maybe a "like" on Facebook or a "favorite" on a tweet?

When building a cohort analysis, it’s important to consider the relationship between the event or interaction you’re tracking and its relationship to your business model.

## Why is it valuable ?

Cohort analysis can be helpful when it comes to understanding your business’ health and "stickiness" - the loyalty of your customers. Stickiness is critical since it’s far cheaper and easier to keep a current customer than to acquire a new one. For startups, it’s also a key indicator of product-market fit.

Additionally, your product evolves over time. New features are added and removed, the design changes, etc. Observing individual groups over time is a starting point to understanding how these changes affect user behavior.

It’s also a good way to visualize your user retention/churn as well as formulating a basic understanding of their lifetime value.

## An example

Imagine we have the following dataset (you can find it [here](https://github.com/synapticielfactory/eland_es_analytics) ):

```python
ed_invoices = ed.read_es(es, 'eland-invoices')
df_invoices = ed.eland_to_pandas(ed_invoices)
df_invoices.head()
```

1. Create a period column based on the OrderDate
Since we're doing monthly cohorts, we'll be looking at the total monthly behavior of our users. Therefore, we don't want granular OrderDate data (right now).

```python
df_invoices['order_period'] = df_invoices.invoice_date.apply(lambda x: x.strftime('%Y-%m'))
df_invoices.head()
```
2. Determine the user's cohort group (based on their first order)

Create a new column called CohortGroup, which is the year and month in which the user's first purchase occurred.

```python
df_invoices.set_index('customer_id', inplace=True)
df_invoices['cohort_group'] = df_invoices.groupby(level=0)['invoice_date'].min().apply(lambda x: x.strftime('%Y-%m'))
df_invoices.reset_index(inplace=True)
df_invoices.head()
```
3. Rollup data by CohortGroup & OrderPeriod

Since we're looking at monthly cohorts, we need to aggregate users, orders, and amount spent by the CohortGroup within the month (OrderPeriod).

```python
grouped = df_invoices.groupby(['cohort_group', 'order_period'])

# count the unique users, orders, and total revenue per Group + Period
cohorts = grouped.agg({'customer_id': pd.Series.nunique,
                       'invoice_id': pd.Series.nunique,
                       'revenue': np.sum})

# make the column names more meaningful
cohorts.rename(columns={'customer_id': 'total_customers',
                        'invoice_id': 'total_orders'}, inplace=True)
cohorts.head()
```

4. Label the CohortPeriod for each CohortGroup

We want to look at how each cohort has behaved in the months following their first purchase, so we'll need to index each cohort to their first purchase month. For example, CohortPeriod = 1 will be the cohort's first month, CohortPeriod = 2 is their second, and so on.

This allows us to compare cohorts across various stages of their lifetime.

```python
def cohort_period(df_invoices):
    """
    Creates a `cohort_period` column, which is the Nth period based on the user's first purchase.
    
    Example
    -------
    Say you want to get the 3rd month for every user:
        df.sort(['customer_id', 'invoice_date', inplace=True)
        df = df.groupby('customer_id').apply(cohort_period)
        df[df.CohortPeriod == 3]
    """
    df_invoices['cohort_range'] = np.arange(len(df_invoices)) + 1
    return df_invoices

cohorts = cohorts.groupby(level=0).apply(cohort_period)
```


## User Retention by Cohort Group

We want to look at the percentage change of each CohortGroup over time -- not the absolute change.

To do this, we'll first need to create a pandas Series containing each CohortGroup and its size.

````python
# reindex the DataFrame
cohorts.reset_index(inplace=True)
cohorts.set_index(['cohort_group', 'cohort_range'], inplace=True)

# create a Series holding the total size of each CohortGroup
cohort_group_size = cohorts['total_customers'].groupby(level=0).first()
cohort_group_size.head(15)
````

Now, we'll need to divide the TotalUsers values in cohorts by cohort_group_size. Since DataFrame operations are performed based on the indices of the objects, we'll use unstack on our cohorts DataFrame to create a matrix where each column represents a CohortGroup and each row is the CohortPeriod corresponding to that group.

To illustrate what unstack does, recall the first five TotalUsers 

````python
cohorts['total_customers'].head(15)
````

And here's what they look like when we unstack the CohortGroup level from the index:

````python
cohorts['total_customers'].unstack(0).head(10)
````

Now, we can utilize broadcasting to divide each column by the corresponding cohort_group_size.

The resulting DataFrame, user_retention, contains the percentage of users from the cohort purchasing within the given period. For instance, 38.4% of users in the 2009-03 purchased again in month 3 (which would be May 2009).

````python
user_retention = cohorts['total_customers'].unstack(0).divide(cohort_group_size, axis=1)
user_retention.head(15)
````

Finally, we can plot the cohorts over time in an effort to spot behavioral differences or similarities. Two common cohort charts are line graphs and heatmaps, both of which are shown below.

Notice that the first period of each cohort is 100% -- this is because our cohorts are based on each user's first purchase, meaning everyone in the cohort purchased in month 1.

````python
user_retention[['2019-01', '2019-02']].plot(figsize=(10,5))

plt.title('Cohorts: User Retention')
plt.xticks(np.arange(1, 12.1, 1))
plt.xlim(1, 12)
plt.ylabel('% of Cohort Purchasing');
````

Let's plot a heatmap

````python
# Creating heatmaps in matplotlib is more difficult than it should be.
# Thankfully, Seaborn makes them easy for us.
# http://stanford.edu/~mwaskom/software/seaborn/

import seaborn as sns
sns.set(style='white')

plt.figure(figsize=(12, 8))
plt.title('Cohorts Analysis : Customers Retention Rate')

sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%');
````

Unsurprisingly, we can see from the above chart that fewer users tend to purchase as time goes on.

However, we can also see that the 2009-01 cohort is the strongest, which enables us to ask targeted questions about this cohort compared to others -- what other attributes (besides first purchase month) do these users share which might be causing them to stick around? How were the majority of these users acquired? Was there a specific marketing campaign that brought them in? Did they take advantage of a promotion at sign-up? The answers to these questions would inform future marketing and product efforts.

## Further work

User retention is only one way of using cohorts to look at your business — we could have also looked at revenue retention. That is, the percentage of each cohort’s month 1 revenue returning in subsequent periods. User retention is important, but we shouldn’t lose sight of the revenue each cohort is bringing in (and how much of it is returning).