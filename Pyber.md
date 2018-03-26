

```python
# Dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```


```python
# Read CSV
city_data = "raw_data/city_data.csv"
ride_data = "raw_data/ride_data.csv"
city_df = pd.read_csv("raw_data/city_data.csv")
ride_df = pd.read_csv("raw_data/ride_data.csv")

# Outer merge the city and ride data on city
combined_data = pd.merge(city_df,ride_df,how='outer', on='city')
combined_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>driver_count</th>
      <th>type</th>
      <th>date</th>
      <th>fare</th>
      <th>ride_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-19 04:27:52</td>
      <td>5.51</td>
      <td>6246006544795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-04-17 06:59:50</td>
      <td>5.54</td>
      <td>7466473222333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-05-04 15:06:07</td>
      <td>30.54</td>
      <td>2140501382736</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-01-25 20:44:56</td>
      <td>12.08</td>
      <td>1896987891309</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-09 18:19:47</td>
      <td>17.91</td>
      <td>8784212854829</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Group data by city
city_df = combined_data.groupby(['city'])
city_df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>driver_count</th>
      <th>type</th>
      <th>date</th>
      <th>fare</th>
      <th>ride_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-19 04:27:52</td>
      <td>5.51</td>
      <td>6246006544795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-04-17 06:59:50</td>
      <td>5.54</td>
      <td>7466473222333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-05-04 15:06:07</td>
      <td>30.54</td>
      <td>2140501382736</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-01-25 20:44:56</td>
      <td>12.08</td>
      <td>1896987891309</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-09 18:19:47</td>
      <td>17.91</td>
      <td>8784212854829</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-10-24 15:15:46</td>
      <td>33.56</td>
      <td>4797969661996</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-06-06 13:54:23</td>
      <td>20.81</td>
      <td>9811478565448</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-10 07:02:40</td>
      <td>44.53</td>
      <td>1563171128434</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-07-05 17:37:13</td>
      <td>29.02</td>
      <td>6897992353955</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-04-25 02:18:31</td>
      <td>20.05</td>
      <td>1148374505062</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Nguyenbury</td>
      <td>8</td>
      <td>Urban</td>
      <td>2016-07-09 04:42:44</td>
      <td>6.28</td>
      <td>1543057793673</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Nguyenbury</td>
      <td>8</td>
      <td>Urban</td>
      <td>2016-11-08 19:22:04</td>
      <td>19.49</td>
      <td>1702803950740</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Nguyenbury</td>
      <td>8</td>
      <td>Urban</td>
      <td>2016-03-19 13:08:09</td>
      <td>35.07</td>
      <td>9198401002936</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Nguyenbury</td>
      <td>8</td>
      <td>Urban</td>
      <td>2016-05-12 15:57:15</td>
      <td>41.63</td>
      <td>224683791660</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Nguyenbury</td>
      <td>8</td>
      <td>Urban</td>
      <td>2016-04-07 06:59:51</td>
      <td>19.01</td>
      <td>4648481871830</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Nguyenbury</td>
      <td>8</td>
      <td>Urban</td>
      <td>2016-08-07 23:33:46</td>
      <td>32.21</td>
      <td>3306944846771</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Nguyenbury</td>
      <td>8</td>
      <td>Urban</td>
      <td>2016-09-20 10:38:34</td>
      <td>42.50</td>
      <td>8781676059272</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Nguyenbury</td>
      <td>8</td>
      <td>Urban</td>
      <td>2016-02-28 06:07:06</td>
      <td>8.41</td>
      <td>8576716154877</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Nguyenbury</td>
      <td>8</td>
      <td>Urban</td>
      <td>2016-09-06 15:59:19</td>
      <td>30.56</td>
      <td>2565292468986</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Nguyenbury</td>
      <td>8</td>
      <td>Urban</td>
      <td>2016-07-16 16:37:52</td>
      <td>26.60</td>
      <td>5120625496769</td>
    </tr>
    <tr>
      <th>54</th>
      <td>East Douglas</td>
      <td>12</td>
      <td>Urban</td>
      <td>2016-10-01 19:07:00</td>
      <td>16.36</td>
      <td>8450340983211</td>
    </tr>
    <tr>
      <th>55</th>
      <td>East Douglas</td>
      <td>12</td>
      <td>Urban</td>
      <td>2016-07-19 07:42:04</td>
      <td>11.24</td>
      <td>8566233760392</td>
    </tr>
    <tr>
      <th>56</th>
      <td>East Douglas</td>
      <td>12</td>
      <td>Urban</td>
      <td>2016-09-20 02:40:41</td>
      <td>23.26</td>
      <td>825335145222</td>
    </tr>
    <tr>
      <th>57</th>
      <td>East Douglas</td>
      <td>12</td>
      <td>Urban</td>
      <td>2016-04-02 13:49:14</td>
      <td>28.17</td>
      <td>3800595642657</td>
    </tr>
    <tr>
      <th>58</th>
      <td>East Douglas</td>
      <td>12</td>
      <td>Urban</td>
      <td>2016-10-19 20:25:16</td>
      <td>28.18</td>
      <td>6204409645686</td>
    </tr>
    <tr>
      <th>59</th>
      <td>East Douglas</td>
      <td>12</td>
      <td>Urban</td>
      <td>2016-09-19 09:23:08</td>
      <td>34.17</td>
      <td>8612766947608</td>
    </tr>
    <tr>
      <th>60</th>
      <td>East Douglas</td>
      <td>12</td>
      <td>Urban</td>
      <td>2016-02-10 03:30:56</td>
      <td>33.08</td>
      <td>3357256703085</td>
    </tr>
    <tr>
      <th>61</th>
      <td>East Douglas</td>
      <td>12</td>
      <td>Urban</td>
      <td>2016-08-21 22:32:08</td>
      <td>7.90</td>
      <td>4788757321451</td>
    </tr>
    <tr>
      <th>62</th>
      <td>East Douglas</td>
      <td>12</td>
      <td>Urban</td>
      <td>2016-11-14 13:10:08</td>
      <td>22.83</td>
      <td>4916688973313</td>
    </tr>
    <tr>
      <th>63</th>
      <td>East Douglas</td>
      <td>12</td>
      <td>Urban</td>
      <td>2016-09-29 14:23:27</td>
      <td>21.49</td>
      <td>9991382143762</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>East Leslie</td>
      <td>9</td>
      <td>Rural</td>
      <td>2016-04-21 18:44:59</td>
      <td>19.26</td>
      <td>5836114186294</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>East Leslie</td>
      <td>9</td>
      <td>Rural</td>
      <td>2016-04-13 04:30:56</td>
      <td>40.47</td>
      <td>7075058703398</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>East Leslie</td>
      <td>9</td>
      <td>Rural</td>
      <td>2016-04-26 02:34:30</td>
      <td>45.80</td>
      <td>9402873395510</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>East Leslie</td>
      <td>9</td>
      <td>Rural</td>
      <td>2016-04-05 18:53:16</td>
      <td>44.78</td>
      <td>6113138249150</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>East Leslie</td>
      <td>9</td>
      <td>Rural</td>
      <td>2016-11-13 10:21:10</td>
      <td>15.71</td>
      <td>7275986542384</td>
    </tr>
    <tr>
      <th>2381</th>
      <td>East Leslie</td>
      <td>9</td>
      <td>Rural</td>
      <td>2016-03-06 06:10:40</td>
      <td>51.32</td>
      <td>6841691147797</td>
    </tr>
    <tr>
      <th>2382</th>
      <td>East Leslie</td>
      <td>9</td>
      <td>Rural</td>
      <td>2016-03-04 10:18:03</td>
      <td>13.43</td>
      <td>8814831098684</td>
    </tr>
    <tr>
      <th>2383</th>
      <td>East Leslie</td>
      <td>9</td>
      <td>Rural</td>
      <td>2016-11-28 09:09:15</td>
      <td>37.76</td>
      <td>804829686137</td>
    </tr>
    <tr>
      <th>2384</th>
      <td>East Leslie</td>
      <td>9</td>
      <td>Rural</td>
      <td>2016-09-08 19:19:38</td>
      <td>30.59</td>
      <td>8211833105097</td>
    </tr>
    <tr>
      <th>2385</th>
      <td>East Leslie</td>
      <td>9</td>
      <td>Rural</td>
      <td>2016-03-02 22:09:34</td>
      <td>36.61</td>
      <td>5500269118478</td>
    </tr>
    <tr>
      <th>2387</th>
      <td>Hernandezshire</td>
      <td>10</td>
      <td>Rural</td>
      <td>2016-02-20 08:17:32</td>
      <td>58.95</td>
      <td>3176534714830</td>
    </tr>
    <tr>
      <th>2388</th>
      <td>Hernandezshire</td>
      <td>10</td>
      <td>Rural</td>
      <td>2016-06-26 20:11:50</td>
      <td>28.78</td>
      <td>6382848462030</td>
    </tr>
    <tr>
      <th>2389</th>
      <td>Hernandezshire</td>
      <td>10</td>
      <td>Rural</td>
      <td>2016-01-24 00:21:35</td>
      <td>30.32</td>
      <td>7342649945759</td>
    </tr>
    <tr>
      <th>2390</th>
      <td>Hernandezshire</td>
      <td>10</td>
      <td>Rural</td>
      <td>2016-03-05 10:40:16</td>
      <td>23.35</td>
      <td>7443355895137</td>
    </tr>
    <tr>
      <th>2391</th>
      <td>Hernandezshire</td>
      <td>10</td>
      <td>Rural</td>
      <td>2016-04-11 04:44:50</td>
      <td>10.41</td>
      <td>9823290002445</td>
    </tr>
    <tr>
      <th>2392</th>
      <td>Hernandezshire</td>
      <td>10</td>
      <td>Rural</td>
      <td>2016-06-26 11:16:28</td>
      <td>26.29</td>
      <td>304182959218</td>
    </tr>
    <tr>
      <th>2393</th>
      <td>Hernandezshire</td>
      <td>10</td>
      <td>Rural</td>
      <td>2016-11-25 20:34:14</td>
      <td>38.45</td>
      <td>2898512024847</td>
    </tr>
    <tr>
      <th>2394</th>
      <td>Hernandezshire</td>
      <td>10</td>
      <td>Rural</td>
      <td>2016-11-20 17:32:37</td>
      <td>26.79</td>
      <td>3095402154397</td>
    </tr>
    <tr>
      <th>2395</th>
      <td>Hernandezshire</td>
      <td>10</td>
      <td>Rural</td>
      <td>2016-02-24 17:30:44</td>
      <td>44.68</td>
      <td>6389115653382</td>
    </tr>
    <tr>
      <th>2396</th>
      <td>Horneland</td>
      <td>8</td>
      <td>Rural</td>
      <td>2016-07-19 10:07:33</td>
      <td>12.63</td>
      <td>8214498891817</td>
    </tr>
    <tr>
      <th>2397</th>
      <td>Horneland</td>
      <td>8</td>
      <td>Rural</td>
      <td>2016-03-22 21:22:20</td>
      <td>31.53</td>
      <td>1797785685674</td>
    </tr>
    <tr>
      <th>2398</th>
      <td>Horneland</td>
      <td>8</td>
      <td>Rural</td>
      <td>2016-01-26 09:38:17</td>
      <td>21.73</td>
      <td>5665544449606</td>
    </tr>
    <tr>
      <th>2399</th>
      <td>Horneland</td>
      <td>8</td>
      <td>Rural</td>
      <td>2016-03-25 02:05:42</td>
      <td>20.04</td>
      <td>5729327140644</td>
    </tr>
    <tr>
      <th>2400</th>
      <td>West Kevintown</td>
      <td>5</td>
      <td>Rural</td>
      <td>2016-11-27 20:12:58</td>
      <td>12.92</td>
      <td>6460741616450</td>
    </tr>
    <tr>
      <th>2401</th>
      <td>West Kevintown</td>
      <td>5</td>
      <td>Rural</td>
      <td>2016-02-19 01:42:58</td>
      <td>11.15</td>
      <td>8622534016726</td>
    </tr>
    <tr>
      <th>2402</th>
      <td>West Kevintown</td>
      <td>5</td>
      <td>Rural</td>
      <td>2016-03-11 09:03:43</td>
      <td>42.13</td>
      <td>4568909568268</td>
    </tr>
    <tr>
      <th>2403</th>
      <td>West Kevintown</td>
      <td>5</td>
      <td>Rural</td>
      <td>2016-06-25 08:04:12</td>
      <td>24.53</td>
      <td>8188407925972</td>
    </tr>
    <tr>
      <th>2404</th>
      <td>West Kevintown</td>
      <td>5</td>
      <td>Rural</td>
      <td>2016-07-24 13:41:23</td>
      <td>11.78</td>
      <td>2001192693573</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>West Kevintown</td>
      <td>5</td>
      <td>Rural</td>
      <td>2016-06-15 19:53:16</td>
      <td>13.50</td>
      <td>9577921579881</td>
    </tr>
    <tr>
      <th>2406</th>
      <td>West Kevintown</td>
      <td>5</td>
      <td>Rural</td>
      <td>2016-02-10 00:50:04</td>
      <td>34.69</td>
      <td>9595491362610</td>
    </tr>
  </tbody>
</table>
<p>1189 rows × 6 columns</p>
</div>




```python
# Average ($) fare per city
fare = round(city_data['fare'].mean(),2)
# Total number of rides per city
rides = city_data['ride_id'].size()
# Number of drivers per city
drivers = city_data['driver_count'] # size of bubble
# City type
city_type = city_data['type'] # color of bubble

```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-91-193a35cb7207> in <module>()
          1 # Average ($) fare per city
    ----> 2 fare = round(city_data['fare'].mean(),2)
          3 # Total number of rides per city
          4 rides = city_data['ride_id'].size()
          5 # Number of drivers per city
    

    TypeError: string indices must be integers



```python
# Create the bubble plot
plt.scatter(rides, fare, marker='o', edgecolors="black", s=drivers, alpha=0.75)
plt.ylim = (0,45)
plt.xlim = (0,45)
plt.show()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-92-36c231a7eb1c> in <module>()
          1 # Create the bubble plot
    ----> 2 plt.scatter(rides, fare, marker='o', edgecolors="black", s=drivers, alpha=0.75)
          3 plt.ylim = (0,45)
          4 plt.xlim = (0,45)
          5 plt.show()
    

    ~\Anaconda3\lib\site-packages\matplotlib\pyplot.py in scatter(x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, edgecolors, hold, data, **kwargs)
       3432                          vmin=vmin, vmax=vmax, alpha=alpha,
       3433                          linewidths=linewidths, verts=verts,
    -> 3434                          edgecolors=edgecolors, data=data, **kwargs)
       3435     finally:
       3436         ax._hold = washold
    

    ~\Anaconda3\lib\site-packages\matplotlib\__init__.py in inner(ax, *args, **kwargs)
       1896                     warnings.warn(msg % (label_namer, func.__name__),
       1897                                   RuntimeWarning, stacklevel=2)
    -> 1898             return func(ax, *args, **kwargs)
       1899         pre_doc = inner.__doc__
       1900         if pre_doc is None:
    

    ~\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py in scatter(self, x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, edgecolors, **kwargs)
       3970                 s = rcParams['lines.markersize'] ** 2.0
       3971 
    -> 3972         s = np.ma.ravel(s)  # This doesn't have to match x, y in size.
       3973 
       3974         # After this block, c_array will be None unless
    

    ~\Anaconda3\lib\site-packages\numpy\ma\core.py in __call__(self, a, *args, **params)
       6511             a, args[0] = args[0], a
       6512 
    -> 6513         marr = asanyarray(a)
       6514         method_name = self.__name__
       6515         method = getattr(type(marr), method_name, None)
    

    ~\Anaconda3\lib\site-packages\numpy\ma\core.py in asanyarray(a, dtype)
       7803     if isinstance(a, MaskedArray) and (dtype is None or dtype == a.dtype):
       7804         return a
    -> 7805     return masked_array(a, dtype=dtype, copy=False, keep_mask=True, subok=True)
       7806 
       7807 
    

    ~\Anaconda3\lib\site-packages\numpy\ma\core.py in __new__(cls, data, mask, dtype, copy, subok, ndmin, fill_value, keep_mask, hard_mask, shrink, order, **options)
       2807         # Process data.
       2808         _data = np.array(data, dtype=dtype, copy=copy,
    -> 2809                          order=order, subok=True, ndmin=ndmin)
       2810         _baseclass = getattr(data, '_baseclass', type(_data))
       2811         # Check that we're not erasing the mask.
    

    ValueError: setting an array element with a sequence



```python
# Create the bubble plot
#import seaborn as sns
#sns.set_style("ticks")
#sizes = [10,20,30,40,50,60,70,80,90,100]
#marker_size = pd.cut(city_data['driver_count]/2, [0,10,20,30,40,50,60,70,80,90,100], labels=sizes)
#sns.lmplot(rides, fare, data=indicator2, hue='type', scatter_kws={'s':marker_size})
```


      File "<ipython-input-24-a86109eb8509>", line 6
        marker_size = pd.cut(city_data['driver_count]/2, [0,10,20,30,40,50,60,70,80,90,100], labels=sizes)
                                                                                                          ^
    SyntaxError: EOL while scanning string literal
    

