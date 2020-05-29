# Task 1: Introduction

---

For this project, we are going to work on evaluating price of houses given the following features:

1. Year of sale of the house
2. The age of the house at the time of sale
3. Distance from city center
4. Number of stores in the locality
5. The latitude
6. The longitude

![Regression](images/regression.png)

Note: This notebook uses `python 3` and these packages: `tensorflow`, `pandas`, `matplotlib`, `scikit-learn`.

## 1.1: Importing Libraries & Helper Functions

First of all, we will need to import some libraries and helper functions. This includes TensorFlow and some utility functions that I've written to save time.


```python
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

#%matplotlib inline
#tf.logging.set_verbosity(tf.logging.ERROR)

print('Libraries imported.')
```

    Libraries imported.
    

# Task 2: Importing the Data

## 2.1: Importing the Data

The dataset is saved in a `data.csv` file. We will use `pandas` to take a look at some of the rows.


```python
df = pd.read_csv('data.csv', names = column_names) 
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>serial</th>
      <th>date</th>
      <th>age</th>
      <th>distance</th>
      <th>stores</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2009</td>
      <td>21</td>
      <td>9</td>
      <td>6</td>
      <td>84</td>
      <td>121</td>
      <td>14264</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2007</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>86</td>
      <td>121</td>
      <td>12032</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2016</td>
      <td>18</td>
      <td>3</td>
      <td>7</td>
      <td>90</td>
      <td>120</td>
      <td>13560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2002</td>
      <td>13</td>
      <td>2</td>
      <td>2</td>
      <td>80</td>
      <td>128</td>
      <td>12029</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2014</td>
      <td>25</td>
      <td>5</td>
      <td>8</td>
      <td>81</td>
      <td>122</td>
      <td>14157</td>
    </tr>
  </tbody>
</table>
</div>



## 2.2: Check Missing Data

It's a good practice to check if the data has any missing values. In real world data, this is quite common and must be taken care of before any data pre-processing or model training.


```python
df.isna().sum()
```




    serial       0
    date         0
    age          0
    distance     0
    stores       0
    latitude     0
    longitude    0
    price        0
    dtype: int64



# Task 3: Data Normalization

## 3.1: Data Normalization

We can make it easier for optimization algorithms to find minimas by normalizing the data before training a model.


```python
df = df.iloc[:,1:]
df_norm = (df - df.mean()) / df.std()
df_norm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>age</th>
      <th>distance</th>
      <th>stores</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.015978</td>
      <td>0.181384</td>
      <td>1.257002</td>
      <td>0.345224</td>
      <td>-0.307212</td>
      <td>-1.260799</td>
      <td>0.350088</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.350485</td>
      <td>-1.319118</td>
      <td>-0.930610</td>
      <td>-0.609312</td>
      <td>0.325301</td>
      <td>-1.260799</td>
      <td>-1.836486</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.298598</td>
      <td>-0.083410</td>
      <td>-0.618094</td>
      <td>0.663402</td>
      <td>1.590328</td>
      <td>-1.576456</td>
      <td>-0.339584</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.266643</td>
      <td>-0.524735</td>
      <td>-0.930610</td>
      <td>-0.927491</td>
      <td>-1.572238</td>
      <td>0.948803</td>
      <td>-1.839425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.932135</td>
      <td>0.534444</td>
      <td>0.006938</td>
      <td>0.981581</td>
      <td>-1.255981</td>
      <td>-0.945141</td>
      <td>0.245266</td>
    </tr>
  </tbody>
</table>
</div>



## 3.2: Convert Label Value

Because we are using normalized values for the labels, we will get the predictions back from a trained model in the same distribution. So, we need to convert the predicted values back to the original distribution if we want predicted prices.


```python
y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)

print(convert_label_value(0.350088))
```

    14263
    

# Task 4: Create Training and Test Sets

## 4.1: Select Features

Make sure to remove the column __price__ from the list of features as it is the label and should not be used as a feature.


```python
X = df_norm.iloc[:, :6]
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>age</th>
      <th>distance</th>
      <th>stores</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.015978</td>
      <td>0.181384</td>
      <td>1.257002</td>
      <td>0.345224</td>
      <td>-0.307212</td>
      <td>-1.260799</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.350485</td>
      <td>-1.319118</td>
      <td>-0.930610</td>
      <td>-0.609312</td>
      <td>0.325301</td>
      <td>-1.260799</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.298598</td>
      <td>-0.083410</td>
      <td>-0.618094</td>
      <td>0.663402</td>
      <td>1.590328</td>
      <td>-1.576456</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.266643</td>
      <td>-0.524735</td>
      <td>-0.930610</td>
      <td>-0.927491</td>
      <td>-1.572238</td>
      <td>0.948803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.932135</td>
      <td>0.534444</td>
      <td>0.006938</td>
      <td>0.981581</td>
      <td>-1.255981</td>
      <td>-0.945141</td>
    </tr>
  </tbody>
</table>
</div>



## 4.2: Select Labels


```python
Y = df_norm.iloc[:, -1]
Y.head()
```




    0    0.350088
    1   -1.836486
    2   -0.339584
    3   -1.839425
    4    0.245266
    Name: price, dtype: float64



## 4.3: Feature and Label Values

We will need to extract just the numeric values for the features and labels as the TensorFlow model will expect just numeric values as input.


```python
X_arr = X.values
Y_arr = Y.values

print('X_arr shape: ', X_arr.shape)
print('Y_arr shape: ', Y_arr.shape)
```

    X_arr shape:  (5000, 6)
    Y_arr shape:  (5000,)
    

## 4.4: Train and Test Split

We will keep some part of the data aside as a __test__ set. The model will not use this set during training and it will be used only for checking the performance of the model in trained and un-trained states. This way, we can make sure that we are going in the right direction with our model training.


```python
X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size = 0.05, shuffle = True, random_state=0)

print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)
```

    X_train shape:  (4750, 6)
    y_train shape:  (4750,)
    X_test shape:  (250, 6)
    y_test shape:  (250,)
    

# Task 5: Create the Model

## 5.1: Create the Model

Let's write a function that returns an untrained model of a certain architecture.


```python
def get_model():
    
    model = Sequential([
        Dense(10, input_shape = (6,), activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(5, activation = 'relu'),
        Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer='adadelta'
    )
    
    return model

model = get_model()
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 10)                70        
    _________________________________________________________________
    dense_1 (Dense)              (None, 20)                220       
    _________________________________________________________________
    dense_2 (Dense)              (None, 5)                 105       
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 6         
    =================================================================
    Total params: 401
    Trainable params: 401
    Non-trainable params: 0
    _________________________________________________________________
    

# Task 6: Model Training

## 6.1: Model Training

We can use an `EarlyStopping` callback from Keras to stop the model training if the validation loss stops decreasing for a few epochs.


```python
early_stopping = EarlyStopping(monitor='val_loss', patience = 5)

model = get_model()

preds_on_untrained = model.predict(X_test)

history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 1000,
    callbacks = [early_stopping]
)
```

    Epoch 1/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.2878 - val_loss: 1.0519
    Epoch 2/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2854 - val_loss: 1.0502
    Epoch 3/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2830 - val_loss: 1.0486
    Epoch 4/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2807 - val_loss: 1.0469
    Epoch 5/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2784 - val_loss: 1.0453
    Epoch 6/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2761 - val_loss: 1.0437
    Epoch 7/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2738 - val_loss: 1.0420
    Epoch 8/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2714 - val_loss: 1.0404
    Epoch 9/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2691 - val_loss: 1.0387
    Epoch 10/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2668 - val_loss: 1.0372
    Epoch 11/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2646 - val_loss: 1.0356
    Epoch 12/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2624 - val_loss: 1.0341
    Epoch 13/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2602 - val_loss: 1.0325
    Epoch 14/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2579 - val_loss: 1.0309
    Epoch 15/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.2557 - val_loss: 1.0293
    Epoch 16/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2535 - val_loss: 1.0278
    Epoch 17/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2513 - val_loss: 1.0263
    Epoch 18/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2492 - val_loss: 1.0248
    Epoch 19/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2471 - val_loss: 1.0233
    Epoch 20/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2450 - val_loss: 1.0219
    Epoch 21/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2430 - val_loss: 1.0204
    Epoch 22/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2409 - val_loss: 1.0190
    Epoch 23/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2388 - val_loss: 1.0175
    Epoch 24/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2367 - val_loss: 1.0161
    Epoch 25/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2347 - val_loss: 1.0146
    Epoch 26/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2327 - val_loss: 1.0132
    Epoch 27/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2307 - val_loss: 1.0119
    Epoch 28/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2288 - val_loss: 1.0105
    Epoch 29/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.2268 - val_loss: 1.0092
    Epoch 30/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.2249 - val_loss: 1.0079
    Epoch 31/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.2230 - val_loss: 1.0065
    Epoch 32/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2211 - val_loss: 1.0052
    Epoch 33/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2193 - val_loss: 1.0039
    Epoch 34/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2174 - val_loss: 1.0026
    Epoch 35/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2155 - val_loss: 1.0012
    Epoch 36/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2137 - val_loss: 0.9999
    Epoch 37/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2118 - val_loss: 0.9986
    Epoch 38/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2100 - val_loss: 0.9973
    Epoch 39/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2081 - val_loss: 0.9961
    Epoch 40/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.2064 - val_loss: 0.9949
    Epoch 41/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2046 - val_loss: 0.9936
    Epoch 42/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2029 - val_loss: 0.9924
    Epoch 43/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.2011 - val_loss: 0.9912
    Epoch 44/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1994 - val_loss: 0.9900
    Epoch 45/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1978 - val_loss: 0.9888
    Epoch 46/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1961 - val_loss: 0.9877
    Epoch 47/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1944 - val_loss: 0.9865
    Epoch 48/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1927 - val_loss: 0.9853
    Epoch 49/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1911 - val_loss: 0.9841
    Epoch 50/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1894 - val_loss: 0.9829
    Epoch 51/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1877 - val_loss: 0.9818
    Epoch 52/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1861 - val_loss: 0.9806
    Epoch 53/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1844 - val_loss: 0.9794
    Epoch 54/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1828 - val_loss: 0.9783
    Epoch 55/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.1812 - val_loss: 0.9771
    Epoch 56/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1796 - val_loss: 0.9760
    Epoch 57/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.1780 - val_loss: 0.9749
    Epoch 58/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1764 - val_loss: 0.9738
    Epoch 59/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1749 - val_loss: 0.9727
    Epoch 60/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1733 - val_loss: 0.9716
    Epoch 61/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1718 - val_loss: 0.9705
    Epoch 62/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1703 - val_loss: 0.9695
    Epoch 63/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1688 - val_loss: 0.9684
    Epoch 64/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1673 - val_loss: 0.9674
    Epoch 65/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1658 - val_loss: 0.9663
    Epoch 66/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1643 - val_loss: 0.9653
    Epoch 67/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1628 - val_loss: 0.9642
    Epoch 68/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1614 - val_loss: 0.9632
    Epoch 69/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1599 - val_loss: 0.9622
    Epoch 70/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1585 - val_loss: 0.9612
    Epoch 71/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1571 - val_loss: 0.9602
    Epoch 72/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1557 - val_loss: 0.9593
    Epoch 73/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1543 - val_loss: 0.9583
    Epoch 74/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.1529 - val_loss: 0.9573
    Epoch 75/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1515 - val_loss: 0.9563
    Epoch 76/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1501 - val_loss: 0.9553
    Epoch 77/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1487 - val_loss: 0.9544
    Epoch 78/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1473 - val_loss: 0.9534
    Epoch 79/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1459 - val_loss: 0.9524
    Epoch 80/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1445 - val_loss: 0.9514
    Epoch 81/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1431 - val_loss: 0.9505
    Epoch 82/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1418 - val_loss: 0.9496
    Epoch 83/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1405 - val_loss: 0.9487
    Epoch 84/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.1392 - val_loss: 0.9478
    Epoch 85/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1379 - val_loss: 0.9469
    Epoch 86/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1366 - val_loss: 0.9460
    Epoch 87/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.1353 - val_loss: 0.9450
    Epoch 88/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1341 - val_loss: 0.9441
    Epoch 89/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1328 - val_loss: 0.9432
    Epoch 90/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1315 - val_loss: 0.9423
    Epoch 91/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1303 - val_loss: 0.9415
    Epoch 92/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1290 - val_loss: 0.9406
    Epoch 93/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1278 - val_loss: 0.9397
    Epoch 94/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1266 - val_loss: 0.9388
    Epoch 95/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1254 - val_loss: 0.9380
    Epoch 96/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.1242 - val_loss: 0.9371
    Epoch 97/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1229 - val_loss: 0.9363
    Epoch 98/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1217 - val_loss: 0.9354
    Epoch 99/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1205 - val_loss: 0.9345
    Epoch 100/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1193 - val_loss: 0.9337
    Epoch 101/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1181 - val_loss: 0.9328
    Epoch 102/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.1169 - val_loss: 0.9320
    Epoch 103/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.1157 - val_loss: 0.9312
    Epoch 104/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1146 - val_loss: 0.9303
    Epoch 105/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1134 - val_loss: 0.9295
    Epoch 106/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1123 - val_loss: 0.9287
    Epoch 107/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.1111 - val_loss: 0.9279
    Epoch 108/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1100 - val_loss: 0.9271
    Epoch 109/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1089 - val_loss: 0.9263
    Epoch 110/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1078 - val_loss: 0.9255
    Epoch 111/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1067 - val_loss: 0.9247
    Epoch 112/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1055 - val_loss: 0.9239
    Epoch 113/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1044 - val_loss: 0.9231
    Epoch 114/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1033 - val_loss: 0.9224
    Epoch 115/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.1023 - val_loss: 0.9216
    Epoch 116/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.1013 - val_loss: 0.9209
    Epoch 117/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.1003 - val_loss: 0.9202
    Epoch 118/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.0992 - val_loss: 0.9195
    Epoch 119/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0983 - val_loss: 0.9188
    Epoch 120/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0973 - val_loss: 0.9181
    Epoch 121/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0963 - val_loss: 0.9174
    Epoch 122/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0954 - val_loss: 0.9167
    Epoch 123/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0944 - val_loss: 0.9160
    Epoch 124/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0935 - val_loss: 0.9153
    Epoch 125/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0925 - val_loss: 0.9146
    Epoch 126/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0915 - val_loss: 0.9139
    Epoch 127/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0905 - val_loss: 0.9132
    Epoch 128/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0896 - val_loss: 0.9125
    Epoch 129/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0886 - val_loss: 0.9118
    Epoch 130/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0877 - val_loss: 0.9111
    Epoch 131/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0868 - val_loss: 0.9105
    Epoch 132/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0858 - val_loss: 0.9098
    Epoch 133/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0849 - val_loss: 0.9091
    Epoch 134/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0839 - val_loss: 0.9084
    Epoch 135/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0830 - val_loss: 0.9077
    Epoch 136/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0821 - val_loss: 0.9071
    Epoch 137/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0812 - val_loss: 0.9064
    Epoch 138/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0803 - val_loss: 0.9057
    Epoch 139/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0793 - val_loss: 0.9051
    Epoch 140/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0784 - val_loss: 0.9045
    Epoch 141/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0776 - val_loss: 0.9038
    Epoch 142/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0767 - val_loss: 0.9032
    Epoch 143/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0758 - val_loss: 0.9026
    Epoch 144/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0750 - val_loss: 0.9020
    Epoch 145/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0741 - val_loss: 0.9014
    Epoch 146/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0733 - val_loss: 0.9007
    Epoch 147/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0724 - val_loss: 0.9001
    Epoch 148/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0715 - val_loss: 0.8995
    Epoch 149/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0707 - val_loss: 0.8989
    Epoch 150/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0699 - val_loss: 0.8983
    Epoch 151/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0690 - val_loss: 0.8977
    Epoch 152/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0682 - val_loss: 0.8971
    Epoch 153/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0674 - val_loss: 0.8966
    Epoch 154/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0665 - val_loss: 0.8960
    Epoch 155/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0657 - val_loss: 0.8954
    Epoch 156/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0649 - val_loss: 0.8948
    Epoch 157/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0641 - val_loss: 0.8942
    Epoch 158/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0633 - val_loss: 0.8937
    Epoch 159/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0624 - val_loss: 0.8931
    Epoch 160/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0616 - val_loss: 0.8925
    Epoch 161/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0609 - val_loss: 0.8920
    Epoch 162/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0601 - val_loss: 0.8914
    Epoch 163/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0593 - val_loss: 0.8909
    Epoch 164/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0585 - val_loss: 0.8903
    Epoch 165/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0578 - val_loss: 0.8898
    Epoch 166/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0570 - val_loss: 0.8892
    Epoch 167/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0563 - val_loss: 0.8887
    Epoch 168/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0555 - val_loss: 0.8882
    Epoch 169/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0548 - val_loss: 0.8876
    Epoch 170/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0540 - val_loss: 0.8871
    Epoch 171/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0533 - val_loss: 0.8866
    Epoch 172/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0525 - val_loss: 0.8860
    Epoch 173/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0518 - val_loss: 0.8855
    Epoch 174/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0511 - val_loss: 0.8850
    Epoch 175/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0504 - val_loss: 0.8845
    Epoch 176/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0497 - val_loss: 0.8840
    Epoch 177/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0489 - val_loss: 0.8834
    Epoch 178/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0482 - val_loss: 0.8829
    Epoch 179/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0475 - val_loss: 0.8824
    Epoch 180/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0468 - val_loss: 0.8819
    Epoch 181/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0461 - val_loss: 0.8814
    Epoch 182/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0454 - val_loss: 0.8809
    Epoch 183/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0447 - val_loss: 0.8805
    Epoch 184/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0440 - val_loss: 0.8800
    Epoch 185/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0433 - val_loss: 0.8795
    Epoch 186/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0427 - val_loss: 0.8790
    Epoch 187/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0420 - val_loss: 0.8785
    Epoch 188/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0413 - val_loss: 0.8781
    Epoch 189/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0406 - val_loss: 0.8776
    Epoch 190/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0400 - val_loss: 0.8771
    Epoch 191/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0393 - val_loss: 0.8767
    Epoch 192/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0387 - val_loss: 0.8762
    Epoch 193/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0380 - val_loss: 0.8757
    Epoch 194/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0373 - val_loss: 0.8753
    Epoch 195/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0367 - val_loss: 0.8748
    Epoch 196/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0360 - val_loss: 0.8744
    Epoch 197/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0354 - val_loss: 0.8739
    Epoch 198/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0348 - val_loss: 0.8735
    Epoch 199/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0341 - val_loss: 0.8730
    Epoch 200/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0335 - val_loss: 0.8726
    Epoch 201/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0329 - val_loss: 0.8722
    Epoch 202/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0322 - val_loss: 0.8717
    Epoch 203/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0316 - val_loss: 0.8713
    Epoch 204/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0310 - val_loss: 0.8708
    Epoch 205/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0303 - val_loss: 0.8704
    Epoch 206/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0297 - val_loss: 0.8699
    Epoch 207/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0291 - val_loss: 0.8695
    Epoch 208/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0285 - val_loss: 0.8691
    Epoch 209/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0279 - val_loss: 0.8687
    Epoch 210/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0273 - val_loss: 0.8682
    Epoch 211/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0267 - val_loss: 0.8678
    Epoch 212/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0261 - val_loss: 0.8674
    Epoch 213/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0255 - val_loss: 0.8670
    Epoch 214/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0249 - val_loss: 0.8666
    Epoch 215/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0243 - val_loss: 0.8661
    Epoch 216/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0237 - val_loss: 0.8657
    Epoch 217/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0232 - val_loss: 0.8653
    Epoch 218/1000
    149/149 [==============================] - 0s 3ms/step - loss: 1.0226 - val_loss: 0.8649
    Epoch 219/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0220 - val_loss: 0.8645
    Epoch 220/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0214 - val_loss: 0.8640
    Epoch 221/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0208 - val_loss: 0.8636
    Epoch 222/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0203 - val_loss: 0.8632
    Epoch 223/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0197 - val_loss: 0.8628
    Epoch 224/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0191 - val_loss: 0.8624
    Epoch 225/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0185 - val_loss: 0.8619
    Epoch 226/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0179 - val_loss: 0.8615
    Epoch 227/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0174 - val_loss: 0.8611
    Epoch 228/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0168 - val_loss: 0.8607
    Epoch 229/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0162 - val_loss: 0.8603
    Epoch 230/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0157 - val_loss: 0.8599
    Epoch 231/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0151 - val_loss: 0.8595
    Epoch 232/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0146 - val_loss: 0.8591
    Epoch 233/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0140 - val_loss: 0.8587
    Epoch 234/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0135 - val_loss: 0.8583
    Epoch 235/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0129 - val_loss: 0.8579
    Epoch 236/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0124 - val_loss: 0.8575
    Epoch 237/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0119 - val_loss: 0.8571
    Epoch 238/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0113 - val_loss: 0.8567
    Epoch 239/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0108 - val_loss: 0.8563
    Epoch 240/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0103 - val_loss: 0.8560
    Epoch 241/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0097 - val_loss: 0.8556
    Epoch 242/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0092 - val_loss: 0.8552
    Epoch 243/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0087 - val_loss: 0.8548
    Epoch 244/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0081 - val_loss: 0.8544
    Epoch 245/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0076 - val_loss: 0.8540
    Epoch 246/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0071 - val_loss: 0.8536
    Epoch 247/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0066 - val_loss: 0.8533
    Epoch 248/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0060 - val_loss: 0.8529
    Epoch 249/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0055 - val_loss: 0.8525
    Epoch 250/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0050 - val_loss: 0.8521
    Epoch 251/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0045 - val_loss: 0.8517
    Epoch 252/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0040 - val_loss: 0.8514
    Epoch 253/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0035 - val_loss: 0.8510
    Epoch 254/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0030 - val_loss: 0.8506
    Epoch 255/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0025 - val_loss: 0.8502
    Epoch 256/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0020 - val_loss: 0.8498
    Epoch 257/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0015 - val_loss: 0.8494
    Epoch 258/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0010 - val_loss: 0.8491
    Epoch 259/1000
    149/149 [==============================] - 0s 2ms/step - loss: 1.0005 - val_loss: 0.8487
    Epoch 260/1000
    149/149 [==============================] - 0s 1ms/step - loss: 1.0000 - val_loss: 0.8483
    Epoch 261/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9995 - val_loss: 0.8479
    Epoch 262/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9990 - val_loss: 0.8475
    Epoch 263/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9985 - val_loss: 0.8471
    Epoch 264/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9980 - val_loss: 0.8467
    Epoch 265/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9975 - val_loss: 0.8463
    Epoch 266/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9970 - val_loss: 0.8459
    Epoch 267/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9965 - val_loss: 0.8455
    Epoch 268/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9960 - val_loss: 0.8451
    Epoch 269/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9955 - val_loss: 0.8447
    Epoch 270/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9950 - val_loss: 0.8443
    Epoch 271/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9945 - val_loss: 0.8439
    Epoch 272/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9940 - val_loss: 0.8435
    Epoch 273/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9935 - val_loss: 0.8432
    Epoch 274/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9930 - val_loss: 0.8428
    Epoch 275/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9925 - val_loss: 0.8424
    Epoch 276/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9921 - val_loss: 0.8420
    Epoch 277/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9916 - val_loss: 0.8416
    Epoch 278/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9911 - val_loss: 0.8412
    Epoch 279/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9906 - val_loss: 0.8408
    Epoch 280/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9901 - val_loss: 0.8404
    Epoch 281/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9896 - val_loss: 0.8400
    Epoch 282/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9891 - val_loss: 0.8396
    Epoch 283/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9887 - val_loss: 0.8392
    Epoch 284/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9882 - val_loss: 0.8388
    Epoch 285/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9877 - val_loss: 0.8384
    Epoch 286/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9872 - val_loss: 0.8380
    Epoch 287/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9867 - val_loss: 0.8376
    Epoch 288/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9863 - val_loss: 0.8372
    Epoch 289/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9858 - val_loss: 0.8368
    Epoch 290/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9853 - val_loss: 0.8364
    Epoch 291/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9848 - val_loss: 0.8360
    Epoch 292/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9843 - val_loss: 0.8356
    Epoch 293/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9839 - val_loss: 0.8352
    Epoch 294/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9834 - val_loss: 0.8348
    Epoch 295/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9829 - val_loss: 0.8344
    Epoch 296/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9824 - val_loss: 0.8340
    Epoch 297/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9819 - val_loss: 0.8336
    Epoch 298/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9815 - val_loss: 0.8331
    Epoch 299/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9810 - val_loss: 0.8327
    Epoch 300/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9805 - val_loss: 0.8323
    Epoch 301/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9800 - val_loss: 0.8319
    Epoch 302/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9795 - val_loss: 0.8315
    Epoch 303/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9790 - val_loss: 0.8311
    Epoch 304/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9786 - val_loss: 0.8307
    Epoch 305/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9781 - val_loss: 0.8303
    Epoch 306/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9776 - val_loss: 0.8298
    Epoch 307/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9771 - val_loss: 0.8294
    Epoch 308/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9767 - val_loss: 0.8290
    Epoch 309/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9762 - val_loss: 0.8286
    Epoch 310/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9757 - val_loss: 0.8282
    Epoch 311/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9752 - val_loss: 0.8278
    Epoch 312/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9748 - val_loss: 0.8274
    Epoch 313/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9743 - val_loss: 0.8269
    Epoch 314/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9738 - val_loss: 0.8265
    Epoch 315/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9733 - val_loss: 0.8261
    Epoch 316/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9728 - val_loss: 0.8257
    Epoch 317/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9723 - val_loss: 0.8252
    Epoch 318/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9719 - val_loss: 0.8248
    Epoch 319/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9714 - val_loss: 0.8244
    Epoch 320/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9709 - val_loss: 0.8239
    Epoch 321/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9704 - val_loss: 0.8235
    Epoch 322/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9700 - val_loss: 0.8231
    Epoch 323/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9695 - val_loss: 0.8227
    Epoch 324/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9690 - val_loss: 0.8223
    Epoch 325/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9685 - val_loss: 0.8218
    Epoch 326/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9681 - val_loss: 0.8214
    Epoch 327/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9676 - val_loss: 0.8210
    Epoch 328/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9671 - val_loss: 0.8206
    Epoch 329/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9666 - val_loss: 0.8201
    Epoch 330/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9662 - val_loss: 0.8197
    Epoch 331/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9657 - val_loss: 0.8193
    Epoch 332/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9652 - val_loss: 0.8189
    Epoch 333/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9647 - val_loss: 0.8184
    Epoch 334/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9643 - val_loss: 0.8180
    Epoch 335/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9638 - val_loss: 0.8176
    Epoch 336/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9633 - val_loss: 0.8171
    Epoch 337/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9628 - val_loss: 0.8167
    Epoch 338/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9623 - val_loss: 0.8163
    Epoch 339/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9618 - val_loss: 0.8158
    Epoch 340/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9614 - val_loss: 0.8154
    Epoch 341/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9609 - val_loss: 0.8149
    Epoch 342/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9604 - val_loss: 0.8145
    Epoch 343/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9599 - val_loss: 0.8141
    Epoch 344/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9595 - val_loss: 0.8136
    Epoch 345/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9590 - val_loss: 0.8132
    Epoch 346/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9585 - val_loss: 0.8128
    Epoch 347/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9580 - val_loss: 0.8123
    Epoch 348/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9575 - val_loss: 0.8119
    Epoch 349/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9570 - val_loss: 0.8114
    Epoch 350/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9566 - val_loss: 0.8110
    Epoch 351/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9561 - val_loss: 0.8105
    Epoch 352/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9556 - val_loss: 0.8101
    Epoch 353/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9551 - val_loss: 0.8096
    Epoch 354/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9546 - val_loss: 0.8092
    Epoch 355/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9541 - val_loss: 0.8087
    Epoch 356/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9536 - val_loss: 0.8083
    Epoch 357/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9531 - val_loss: 0.8078
    Epoch 358/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9526 - val_loss: 0.8074
    Epoch 359/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9521 - val_loss: 0.8069
    Epoch 360/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9516 - val_loss: 0.8065
    Epoch 361/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9512 - val_loss: 0.8060
    Epoch 362/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9507 - val_loss: 0.8056
    Epoch 363/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9502 - val_loss: 0.8051
    Epoch 364/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9497 - val_loss: 0.8046
    Epoch 365/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9492 - val_loss: 0.8042
    Epoch 366/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9487 - val_loss: 0.8037
    Epoch 367/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9482 - val_loss: 0.8033
    Epoch 368/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9477 - val_loss: 0.8028
    Epoch 369/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9472 - val_loss: 0.8024
    Epoch 370/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9467 - val_loss: 0.8019
    Epoch 371/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9462 - val_loss: 0.8015
    Epoch 372/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9457 - val_loss: 0.8010
    Epoch 373/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9452 - val_loss: 0.8006
    Epoch 374/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9447 - val_loss: 0.8001
    Epoch 375/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9442 - val_loss: 0.7997
    Epoch 376/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9437 - val_loss: 0.7992
    Epoch 377/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9432 - val_loss: 0.7988
    Epoch 378/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9427 - val_loss: 0.7983
    Epoch 379/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9422 - val_loss: 0.7979
    Epoch 380/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9417 - val_loss: 0.7974
    Epoch 381/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9412 - val_loss: 0.7970
    Epoch 382/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9407 - val_loss: 0.7965
    Epoch 383/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9402 - val_loss: 0.7960
    Epoch 384/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9397 - val_loss: 0.7956
    Epoch 385/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9392 - val_loss: 0.7951
    Epoch 386/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.9387 - val_loss: 0.7947
    Epoch 387/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.9382 - val_loss: 0.7942
    Epoch 388/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.9377 - val_loss: 0.7938
    Epoch 389/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9372 - val_loss: 0.7933
    Epoch 390/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9367 - val_loss: 0.7929
    Epoch 391/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9362 - val_loss: 0.7924
    Epoch 392/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9357 - val_loss: 0.7919
    Epoch 393/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9352 - val_loss: 0.7915
    Epoch 394/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9347 - val_loss: 0.7910
    Epoch 395/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9342 - val_loss: 0.7905
    Epoch 396/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9336 - val_loss: 0.7900
    Epoch 397/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9331 - val_loss: 0.7896
    Epoch 398/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9326 - val_loss: 0.7891
    Epoch 399/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9321 - val_loss: 0.7886
    Epoch 400/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9315 - val_loss: 0.7881
    Epoch 401/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9310 - val_loss: 0.7877
    Epoch 402/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9305 - val_loss: 0.7872
    Epoch 403/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9300 - val_loss: 0.7867
    Epoch 404/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9295 - val_loss: 0.7862
    Epoch 405/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9290 - val_loss: 0.7858
    Epoch 406/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9285 - val_loss: 0.7853
    Epoch 407/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9280 - val_loss: 0.7848
    Epoch 408/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9275 - val_loss: 0.7844
    Epoch 409/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9270 - val_loss: 0.7839
    Epoch 410/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9264 - val_loss: 0.7834
    Epoch 411/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9259 - val_loss: 0.7829
    Epoch 412/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9254 - val_loss: 0.7825
    Epoch 413/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9249 - val_loss: 0.7820
    Epoch 414/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9244 - val_loss: 0.7815
    Epoch 415/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9238 - val_loss: 0.7810
    Epoch 416/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9233 - val_loss: 0.7805
    Epoch 417/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9228 - val_loss: 0.7801
    Epoch 418/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9223 - val_loss: 0.7796
    Epoch 419/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9218 - val_loss: 0.7791
    Epoch 420/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9212 - val_loss: 0.7786
    Epoch 421/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9207 - val_loss: 0.7781
    Epoch 422/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9202 - val_loss: 0.7776
    Epoch 423/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9197 - val_loss: 0.7771
    Epoch 424/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9191 - val_loss: 0.7766
    Epoch 425/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9186 - val_loss: 0.7761
    Epoch 426/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9181 - val_loss: 0.7756
    Epoch 427/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9175 - val_loss: 0.7751
    Epoch 428/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9170 - val_loss: 0.7746
    Epoch 429/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9165 - val_loss: 0.7741
    Epoch 430/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9159 - val_loss: 0.7736
    Epoch 431/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9154 - val_loss: 0.7731
    Epoch 432/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9149 - val_loss: 0.7726
    Epoch 433/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9143 - val_loss: 0.7721
    Epoch 434/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9138 - val_loss: 0.7716
    Epoch 435/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9132 - val_loss: 0.7711
    Epoch 436/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9127 - val_loss: 0.7706
    Epoch 437/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9122 - val_loss: 0.7701
    Epoch 438/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9116 - val_loss: 0.7696
    Epoch 439/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9111 - val_loss: 0.7690
    Epoch 440/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9105 - val_loss: 0.7685
    Epoch 441/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9100 - val_loss: 0.7680
    Epoch 442/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9095 - val_loss: 0.7675
    Epoch 443/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9089 - val_loss: 0.7670
    Epoch 444/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9084 - val_loss: 0.7664
    Epoch 445/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9078 - val_loss: 0.7659
    Epoch 446/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9073 - val_loss: 0.7654
    Epoch 447/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9067 - val_loss: 0.7649
    Epoch 448/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9062 - val_loss: 0.7643
    Epoch 449/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9056 - val_loss: 0.7638
    Epoch 450/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9051 - val_loss: 0.7633
    Epoch 451/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9045 - val_loss: 0.7627
    Epoch 452/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9040 - val_loss: 0.7622
    Epoch 453/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9034 - val_loss: 0.7617
    Epoch 454/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9029 - val_loss: 0.7612
    Epoch 455/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9024 - val_loss: 0.7607
    Epoch 456/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9018 - val_loss: 0.7601
    Epoch 457/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9013 - val_loss: 0.7596
    Epoch 458/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.9007 - val_loss: 0.7591
    Epoch 459/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.9002 - val_loss: 0.7585
    Epoch 460/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8996 - val_loss: 0.7580
    Epoch 461/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8990 - val_loss: 0.7574
    Epoch 462/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8985 - val_loss: 0.7569
    Epoch 463/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8979 - val_loss: 0.7563
    Epoch 464/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8973 - val_loss: 0.7558
    Epoch 465/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8968 - val_loss: 0.7552
    Epoch 466/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8962 - val_loss: 0.7547
    Epoch 467/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8956 - val_loss: 0.7541
    Epoch 468/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8951 - val_loss: 0.7536
    Epoch 469/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8945 - val_loss: 0.7530
    Epoch 470/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8939 - val_loss: 0.7525
    Epoch 471/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8934 - val_loss: 0.7519
    Epoch 472/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8928 - val_loss: 0.7513
    Epoch 473/1000
    149/149 [==============================] - ETA: 0s - loss: 0.888 - 0s 2ms/step - loss: 0.8922 - val_loss: 0.7508
    Epoch 474/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8916 - val_loss: 0.7502
    Epoch 475/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8911 - val_loss: 0.7496
    Epoch 476/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8905 - val_loss: 0.7491
    Epoch 477/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8899 - val_loss: 0.7485
    Epoch 478/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8893 - val_loss: 0.7480
    Epoch 479/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8888 - val_loss: 0.7474
    Epoch 480/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8882 - val_loss: 0.7468
    Epoch 481/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8876 - val_loss: 0.7463
    Epoch 482/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8870 - val_loss: 0.7457
    Epoch 483/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8865 - val_loss: 0.7451
    Epoch 484/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8859 - val_loss: 0.7446
    Epoch 485/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8853 - val_loss: 0.7440
    Epoch 486/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8847 - val_loss: 0.7435
    Epoch 487/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8842 - val_loss: 0.7429
    Epoch 488/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8836 - val_loss: 0.7423
    Epoch 489/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8830 - val_loss: 0.7418
    Epoch 490/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8824 - val_loss: 0.7412
    Epoch 491/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8819 - val_loss: 0.7407
    Epoch 492/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8813 - val_loss: 0.7401
    Epoch 493/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8807 - val_loss: 0.7395
    Epoch 494/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8802 - val_loss: 0.7390
    Epoch 495/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8796 - val_loss: 0.7384
    Epoch 496/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8790 - val_loss: 0.7378
    Epoch 497/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8784 - val_loss: 0.7373
    Epoch 498/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8778 - val_loss: 0.7367
    Epoch 499/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8772 - val_loss: 0.7361
    Epoch 500/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8767 - val_loss: 0.7356
    Epoch 501/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8761 - val_loss: 0.7350
    Epoch 502/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8755 - val_loss: 0.7344
    Epoch 503/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8749 - val_loss: 0.7339
    Epoch 504/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8744 - val_loss: 0.7333
    Epoch 505/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8738 - val_loss: 0.7327
    Epoch 506/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8732 - val_loss: 0.7321
    Epoch 507/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8726 - val_loss: 0.7315
    Epoch 508/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8720 - val_loss: 0.7310
    Epoch 509/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8714 - val_loss: 0.7304
    Epoch 510/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8708 - val_loss: 0.7298
    Epoch 511/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8702 - val_loss: 0.7292
    Epoch 512/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8696 - val_loss: 0.7286
    Epoch 513/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8690 - val_loss: 0.7280
    Epoch 514/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8684 - val_loss: 0.7275
    Epoch 515/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8678 - val_loss: 0.7269
    Epoch 516/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8672 - val_loss: 0.7263
    Epoch 517/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8666 - val_loss: 0.7257
    Epoch 518/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8660 - val_loss: 0.7251
    Epoch 519/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8654 - val_loss: 0.7245
    Epoch 520/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8648 - val_loss: 0.7240
    Epoch 521/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8643 - val_loss: 0.7234
    Epoch 522/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8637 - val_loss: 0.7228
    Epoch 523/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8631 - val_loss: 0.7222
    Epoch 524/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8625 - val_loss: 0.7217
    Epoch 525/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8619 - val_loss: 0.7211
    Epoch 526/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8613 - val_loss: 0.7205
    Epoch 527/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.8607 - val_loss: 0.7199
    Epoch 528/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8601 - val_loss: 0.7193
    Epoch 529/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8595 - val_loss: 0.7187
    Epoch 530/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8589 - val_loss: 0.7181
    Epoch 531/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8583 - val_loss: 0.7175
    Epoch 532/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8577 - val_loss: 0.7169
    Epoch 533/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8571 - val_loss: 0.7163
    Epoch 534/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8564 - val_loss: 0.7157
    Epoch 535/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.8558 - val_loss: 0.7151
    Epoch 536/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.8552 - val_loss: 0.7145
    Epoch 537/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.8546 - val_loss: 0.7139
    Epoch 538/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.8540 - val_loss: 0.7133
    Epoch 539/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.8534 - val_loss: 0.7127
    Epoch 540/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.8528 - val_loss: 0.7122
    Epoch 541/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8522 - val_loss: 0.7116
    Epoch 542/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8516 - val_loss: 0.7110
    Epoch 543/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8510 - val_loss: 0.7104
    Epoch 544/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8504 - val_loss: 0.7098
    Epoch 545/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8498 - val_loss: 0.7092
    Epoch 546/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8492 - val_loss: 0.7086
    Epoch 547/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8485 - val_loss: 0.7079
    Epoch 548/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8479 - val_loss: 0.7073
    Epoch 549/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8473 - val_loss: 0.7067
    Epoch 550/1000
    149/149 [==============================] - ETA: 0s - loss: 0.848 - 0s 2ms/step - loss: 0.8467 - val_loss: 0.7061
    Epoch 551/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8461 - val_loss: 0.7055
    Epoch 552/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8454 - val_loss: 0.7049
    Epoch 553/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8448 - val_loss: 0.7043
    Epoch 554/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8442 - val_loss: 0.7037
    Epoch 555/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8436 - val_loss: 0.7031
    Epoch 556/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8430 - val_loss: 0.7025
    Epoch 557/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8423 - val_loss: 0.7018
    Epoch 558/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8417 - val_loss: 0.7012
    Epoch 559/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8411 - val_loss: 0.7006
    Epoch 560/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8405 - val_loss: 0.7000
    Epoch 561/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8398 - val_loss: 0.6994
    Epoch 562/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8392 - val_loss: 0.6988
    Epoch 563/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8386 - val_loss: 0.6982
    Epoch 564/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8380 - val_loss: 0.6976
    Epoch 565/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8374 - val_loss: 0.6969
    Epoch 566/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8368 - val_loss: 0.6963
    Epoch 567/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8361 - val_loss: 0.6957
    Epoch 568/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8355 - val_loss: 0.6951
    Epoch 569/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8349 - val_loss: 0.6945
    Epoch 570/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8343 - val_loss: 0.6939
    Epoch 571/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8337 - val_loss: 0.6933
    Epoch 572/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8330 - val_loss: 0.6927
    Epoch 573/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8324 - val_loss: 0.6921
    Epoch 574/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8318 - val_loss: 0.6915
    Epoch 575/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8312 - val_loss: 0.6908
    Epoch 576/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8306 - val_loss: 0.6902
    Epoch 577/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8299 - val_loss: 0.6896
    Epoch 578/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8293 - val_loss: 0.6890
    Epoch 579/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8287 - val_loss: 0.6884
    Epoch 580/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8280 - val_loss: 0.6877
    Epoch 581/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8274 - val_loss: 0.6871
    Epoch 582/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8268 - val_loss: 0.6865
    Epoch 583/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8261 - val_loss: 0.6858
    Epoch 584/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8255 - val_loss: 0.6852
    Epoch 585/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8248 - val_loss: 0.6846
    Epoch 586/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8242 - val_loss: 0.6839
    Epoch 587/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8235 - val_loss: 0.6833
    Epoch 588/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8229 - val_loss: 0.6827
    Epoch 589/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8222 - val_loss: 0.6820
    Epoch 590/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8216 - val_loss: 0.6814
    Epoch 591/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8209 - val_loss: 0.6807
    Epoch 592/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.8203 - val_loss: 0.6801
    Epoch 593/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8196 - val_loss: 0.6795
    Epoch 594/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8190 - val_loss: 0.6788
    Epoch 595/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8183 - val_loss: 0.6782
    Epoch 596/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8177 - val_loss: 0.6776
    Epoch 597/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8171 - val_loss: 0.6769
    Epoch 598/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8164 - val_loss: 0.6763
    Epoch 599/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8158 - val_loss: 0.6757
    Epoch 600/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8152 - val_loss: 0.6751
    Epoch 601/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8145 - val_loss: 0.6744
    Epoch 602/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8139 - val_loss: 0.6738
    Epoch 603/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8132 - val_loss: 0.6732
    Epoch 604/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8126 - val_loss: 0.6725
    Epoch 605/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8120 - val_loss: 0.6719
    Epoch 606/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8113 - val_loss: 0.6713
    Epoch 607/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8107 - val_loss: 0.6706
    Epoch 608/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8100 - val_loss: 0.6700
    Epoch 609/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8094 - val_loss: 0.6694
    Epoch 610/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8087 - val_loss: 0.6687
    Epoch 611/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8081 - val_loss: 0.6681
    Epoch 612/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8075 - val_loss: 0.6675
    Epoch 613/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8068 - val_loss: 0.6668
    Epoch 614/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8062 - val_loss: 0.6662
    Epoch 615/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8055 - val_loss: 0.6656
    Epoch 616/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8049 - val_loss: 0.6649
    Epoch 617/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8042 - val_loss: 0.6643
    Epoch 618/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8036 - val_loss: 0.6637
    Epoch 619/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8029 - val_loss: 0.6630
    Epoch 620/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8023 - val_loss: 0.6624
    Epoch 621/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8016 - val_loss: 0.6618
    Epoch 622/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8010 - val_loss: 0.6611
    Epoch 623/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.8003 - val_loss: 0.6605
    Epoch 624/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7997 - val_loss: 0.6598
    Epoch 625/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7990 - val_loss: 0.6592
    Epoch 626/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7984 - val_loss: 0.6586
    Epoch 627/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7977 - val_loss: 0.6579
    Epoch 628/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7971 - val_loss: 0.6573
    Epoch 629/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7964 - val_loss: 0.6567
    Epoch 630/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7958 - val_loss: 0.6560
    Epoch 631/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7951 - val_loss: 0.6553
    Epoch 632/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7944 - val_loss: 0.6547
    Epoch 633/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7938 - val_loss: 0.6541
    Epoch 634/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7931 - val_loss: 0.6534
    Epoch 635/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7924 - val_loss: 0.6528
    Epoch 636/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7918 - val_loss: 0.6521
    Epoch 637/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7911 - val_loss: 0.6515
    Epoch 638/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7905 - val_loss: 0.6508
    Epoch 639/1000
    149/149 [==============================] - 1s 4ms/step - loss: 0.7898 - val_loss: 0.6502
    Epoch 640/1000
    149/149 [==============================] - 1s 4ms/step - loss: 0.7891 - val_loss: 0.6495
    Epoch 641/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7884 - val_loss: 0.6489
    Epoch 642/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7878 - val_loss: 0.6482
    Epoch 643/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7871 - val_loss: 0.6476
    Epoch 644/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7865 - val_loss: 0.6469
    Epoch 645/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7858 - val_loss: 0.6463
    Epoch 646/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7852 - val_loss: 0.6457
    Epoch 647/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7845 - val_loss: 0.6450
    Epoch 648/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7838 - val_loss: 0.6444
    Epoch 649/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7832 - val_loss: 0.6437
    Epoch 650/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7825 - val_loss: 0.6431
    Epoch 651/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7819 - val_loss: 0.6424
    Epoch 652/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7812 - val_loss: 0.6418
    Epoch 653/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7805 - val_loss: 0.6411
    Epoch 654/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7798 - val_loss: 0.6405
    Epoch 655/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7792 - val_loss: 0.6398
    Epoch 656/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7785 - val_loss: 0.6392
    Epoch 657/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7778 - val_loss: 0.6385
    Epoch 658/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7771 - val_loss: 0.6379
    Epoch 659/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7765 - val_loss: 0.6372
    Epoch 660/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7758 - val_loss: 0.6366
    Epoch 661/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7751 - val_loss: 0.6359
    Epoch 662/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7745 - val_loss: 0.6353
    Epoch 663/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7738 - val_loss: 0.6347
    Epoch 664/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7732 - val_loss: 0.6340
    Epoch 665/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7725 - val_loss: 0.6334
    Epoch 666/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7719 - val_loss: 0.6328
    Epoch 667/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7712 - val_loss: 0.6321
    Epoch 668/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7705 - val_loss: 0.6315
    Epoch 669/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7699 - val_loss: 0.6308
    Epoch 670/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7692 - val_loss: 0.6302
    Epoch 671/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7685 - val_loss: 0.6296
    Epoch 672/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7679 - val_loss: 0.6289
    Epoch 673/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7672 - val_loss: 0.6283
    Epoch 674/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7666 - val_loss: 0.6276
    Epoch 675/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7659 - val_loss: 0.6270
    Epoch 676/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7652 - val_loss: 0.6264
    Epoch 677/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7646 - val_loss: 0.6257
    Epoch 678/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7639 - val_loss: 0.6251
    Epoch 679/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7632 - val_loss: 0.6244
    Epoch 680/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7625 - val_loss: 0.6238
    Epoch 681/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7619 - val_loss: 0.6231
    Epoch 682/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7612 - val_loss: 0.6225
    Epoch 683/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7606 - val_loss: 0.6219
    Epoch 684/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7599 - val_loss: 0.6212
    Epoch 685/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7592 - val_loss: 0.6206
    Epoch 686/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7586 - val_loss: 0.6199
    Epoch 687/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7579 - val_loss: 0.6193
    Epoch 688/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7572 - val_loss: 0.6186
    Epoch 689/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7565 - val_loss: 0.6180
    Epoch 690/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7558 - val_loss: 0.6173
    Epoch 691/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7551 - val_loss: 0.6166
    Epoch 692/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7545 - val_loss: 0.6160
    Epoch 693/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7538 - val_loss: 0.6153
    Epoch 694/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7531 - val_loss: 0.6147
    Epoch 695/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7524 - val_loss: 0.6140
    Epoch 696/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7518 - val_loss: 0.6134
    Epoch 697/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7511 - val_loss: 0.6128
    Epoch 698/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7504 - val_loss: 0.6121
    Epoch 699/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7497 - val_loss: 0.6114
    Epoch 700/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7491 - val_loss: 0.6108
    Epoch 701/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7484 - val_loss: 0.6101
    Epoch 702/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7477 - val_loss: 0.6095
    Epoch 703/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7470 - val_loss: 0.6088
    Epoch 704/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7463 - val_loss: 0.6082
    Epoch 705/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7457 - val_loss: 0.6075
    Epoch 706/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7450 - val_loss: 0.6069
    Epoch 707/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7443 - val_loss: 0.6062
    Epoch 708/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7437 - val_loss: 0.6056
    Epoch 709/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7430 - val_loss: 0.6049
    Epoch 710/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7423 - val_loss: 0.6043
    Epoch 711/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7416 - val_loss: 0.6037
    Epoch 712/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7410 - val_loss: 0.6030
    Epoch 713/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7403 - val_loss: 0.6024
    Epoch 714/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7397 - val_loss: 0.6018
    Epoch 715/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7390 - val_loss: 0.6011
    Epoch 716/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7383 - val_loss: 0.6004
    Epoch 717/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7376 - val_loss: 0.5998
    Epoch 718/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7370 - val_loss: 0.5991
    Epoch 719/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7363 - val_loss: 0.5985
    Epoch 720/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7356 - val_loss: 0.5978
    Epoch 721/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7350 - val_loss: 0.5972
    Epoch 722/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7343 - val_loss: 0.5965
    Epoch 723/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7336 - val_loss: 0.5959
    Epoch 724/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7330 - val_loss: 0.5953
    Epoch 725/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7323 - val_loss: 0.5946
    Epoch 726/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7316 - val_loss: 0.5940
    Epoch 727/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7309 - val_loss: 0.5933
    Epoch 728/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7303 - val_loss: 0.5926
    Epoch 729/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7296 - val_loss: 0.5920
    Epoch 730/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.7289 - val_loss: 0.5914
    Epoch 731/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7283 - val_loss: 0.5907
    Epoch 732/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7276 - val_loss: 0.5900
    Epoch 733/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7269 - val_loss: 0.5894
    Epoch 734/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7262 - val_loss: 0.5887
    Epoch 735/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7256 - val_loss: 0.5881
    Epoch 736/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7249 - val_loss: 0.5875
    Epoch 737/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7242 - val_loss: 0.5868
    Epoch 738/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7236 - val_loss: 0.5862
    Epoch 739/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7229 - val_loss: 0.5855
    Epoch 740/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7223 - val_loss: 0.5849
    Epoch 741/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7216 - val_loss: 0.5843
    Epoch 742/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7209 - val_loss: 0.5836
    Epoch 743/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7203 - val_loss: 0.5830
    Epoch 744/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7196 - val_loss: 0.5824
    Epoch 745/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7190 - val_loss: 0.5817
    Epoch 746/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7183 - val_loss: 0.5811
    Epoch 747/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7176 - val_loss: 0.5804
    Epoch 748/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7170 - val_loss: 0.5798
    Epoch 749/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7163 - val_loss: 0.5792
    Epoch 750/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7157 - val_loss: 0.5785
    Epoch 751/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7150 - val_loss: 0.5779
    Epoch 752/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7143 - val_loss: 0.5772
    Epoch 753/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7136 - val_loss: 0.5766
    Epoch 754/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7129 - val_loss: 0.5759
    Epoch 755/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7122 - val_loss: 0.5752
    Epoch 756/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7116 - val_loss: 0.5746
    Epoch 757/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7109 - val_loss: 0.5739
    Epoch 758/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7102 - val_loss: 0.5733
    Epoch 759/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7095 - val_loss: 0.5726
    Epoch 760/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7089 - val_loss: 0.5720
    Epoch 761/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7082 - val_loss: 0.5714
    Epoch 762/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7076 - val_loss: 0.5708
    Epoch 763/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7070 - val_loss: 0.5702
    Epoch 764/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7063 - val_loss: 0.5695
    Epoch 765/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7056 - val_loss: 0.5689
    Epoch 766/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7050 - val_loss: 0.5682
    Epoch 767/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7043 - val_loss: 0.5676
    Epoch 768/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7036 - val_loss: 0.5669
    Epoch 769/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7029 - val_loss: 0.5663
    Epoch 770/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7023 - val_loss: 0.5657
    Epoch 771/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.7016 - val_loss: 0.5650
    Epoch 772/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7009 - val_loss: 0.5644
    Epoch 773/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.7003 - val_loss: 0.5637
    Epoch 774/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6996 - val_loss: 0.5631
    Epoch 775/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6989 - val_loss: 0.5625
    Epoch 776/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6983 - val_loss: 0.5618
    Epoch 777/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6976 - val_loss: 0.5612
    Epoch 778/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6970 - val_loss: 0.5606
    Epoch 779/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6963 - val_loss: 0.5599
    Epoch 780/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6956 - val_loss: 0.5593
    Epoch 781/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6950 - val_loss: 0.5586
    Epoch 782/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6943 - val_loss: 0.5580
    Epoch 783/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6936 - val_loss: 0.5574
    Epoch 784/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6930 - val_loss: 0.5567
    Epoch 785/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6923 - val_loss: 0.5561
    Epoch 786/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6917 - val_loss: 0.5555
    Epoch 787/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6910 - val_loss: 0.5549
    Epoch 788/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6904 - val_loss: 0.5542
    Epoch 789/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6897 - val_loss: 0.5536
    Epoch 790/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6891 - val_loss: 0.5530
    Epoch 791/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6884 - val_loss: 0.5524
    Epoch 792/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6878 - val_loss: 0.5518
    Epoch 793/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6872 - val_loss: 0.5512
    Epoch 794/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6865 - val_loss: 0.5506
    Epoch 795/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6859 - val_loss: 0.5500
    Epoch 796/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6852 - val_loss: 0.5494
    Epoch 797/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6846 - val_loss: 0.5488
    Epoch 798/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6840 - val_loss: 0.5481
    Epoch 799/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6833 - val_loss: 0.5475
    Epoch 800/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6827 - val_loss: 0.5469
    Epoch 801/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6821 - val_loss: 0.5463
    Epoch 802/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6814 - val_loss: 0.5457
    Epoch 803/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6808 - val_loss: 0.5451
    Epoch 804/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6802 - val_loss: 0.5445
    Epoch 805/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6795 - val_loss: 0.5439
    Epoch 806/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6789 - val_loss: 0.5433
    Epoch 807/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6782 - val_loss: 0.5427
    Epoch 808/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6776 - val_loss: 0.5421
    Epoch 809/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6770 - val_loss: 0.5415
    Epoch 810/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6764 - val_loss: 0.5409
    Epoch 811/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6757 - val_loss: 0.5404
    Epoch 812/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6751 - val_loss: 0.5398
    Epoch 813/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6745 - val_loss: 0.5392
    Epoch 814/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6739 - val_loss: 0.5386
    Epoch 815/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6732 - val_loss: 0.5380
    Epoch 816/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6726 - val_loss: 0.5374
    Epoch 817/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6719 - val_loss: 0.5368
    Epoch 818/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6713 - val_loss: 0.5362
    Epoch 819/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6706 - val_loss: 0.5356
    Epoch 820/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6700 - val_loss: 0.5350
    Epoch 821/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6694 - val_loss: 0.5344
    Epoch 822/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6687 - val_loss: 0.5338
    Epoch 823/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6681 - val_loss: 0.5332
    Epoch 824/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6675 - val_loss: 0.5326
    Epoch 825/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6669 - val_loss: 0.5321
    Epoch 826/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6662 - val_loss: 0.5315
    Epoch 827/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6656 - val_loss: 0.5309
    Epoch 828/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6650 - val_loss: 0.5303
    Epoch 829/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6643 - val_loss: 0.5297
    Epoch 830/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6637 - val_loss: 0.5291
    Epoch 831/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6631 - val_loss: 0.5286
    Epoch 832/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6625 - val_loss: 0.5280
    Epoch 833/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6618 - val_loss: 0.5274
    Epoch 834/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6612 - val_loss: 0.5268
    Epoch 835/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6606 - val_loss: 0.5262
    Epoch 836/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6600 - val_loss: 0.5257
    Epoch 837/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6593 - val_loss: 0.5251
    Epoch 838/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6587 - val_loss: 0.5245
    Epoch 839/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6581 - val_loss: 0.5239
    Epoch 840/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6574 - val_loss: 0.5233
    Epoch 841/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6568 - val_loss: 0.5228
    Epoch 842/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6562 - val_loss: 0.5222
    Epoch 843/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6556 - val_loss: 0.5216
    Epoch 844/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6550 - val_loss: 0.5210
    Epoch 845/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6544 - val_loss: 0.5205
    Epoch 846/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6538 - val_loss: 0.5199
    Epoch 847/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6531 - val_loss: 0.5193
    Epoch 848/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6525 - val_loss: 0.5187
    Epoch 849/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6519 - val_loss: 0.5182
    Epoch 850/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6513 - val_loss: 0.5176
    Epoch 851/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6507 - val_loss: 0.5170
    Epoch 852/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6501 - val_loss: 0.5164
    Epoch 853/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6494 - val_loss: 0.5159
    Epoch 854/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6488 - val_loss: 0.5153
    Epoch 855/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6482 - val_loss: 0.5147
    Epoch 856/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6476 - val_loss: 0.5141
    Epoch 857/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6469 - val_loss: 0.5135
    Epoch 858/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6463 - val_loss: 0.5129
    Epoch 859/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6457 - val_loss: 0.5124
    Epoch 860/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6451 - val_loss: 0.5118
    Epoch 861/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6445 - val_loss: 0.5113
    Epoch 862/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6439 - val_loss: 0.5107
    Epoch 863/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6433 - val_loss: 0.5102
    Epoch 864/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6427 - val_loss: 0.5096
    Epoch 865/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6422 - val_loss: 0.5091
    Epoch 866/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6416 - val_loss: 0.5085
    Epoch 867/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6410 - val_loss: 0.5080
    Epoch 868/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6404 - val_loss: 0.5074
    Epoch 869/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6398 - val_loss: 0.5068
    Epoch 870/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6392 - val_loss: 0.5063
    Epoch 871/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6386 - val_loss: 0.5057
    Epoch 872/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6380 - val_loss: 0.5052
    Epoch 873/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6374 - val_loss: 0.5046
    Epoch 874/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6368 - val_loss: 0.5041
    Epoch 875/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6362 - val_loss: 0.5035
    Epoch 876/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6357 - val_loss: 0.5030
    Epoch 877/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6351 - val_loss: 0.5024
    Epoch 878/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6345 - val_loss: 0.5019
    Epoch 879/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6339 - val_loss: 0.5014
    Epoch 880/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6333 - val_loss: 0.5008
    Epoch 881/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6328 - val_loss: 0.5003
    Epoch 882/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6322 - val_loss: 0.4997
    Epoch 883/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6316 - val_loss: 0.4992
    Epoch 884/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6310 - val_loss: 0.4987
    Epoch 885/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6305 - val_loss: 0.4981
    Epoch 886/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6299 - val_loss: 0.4976
    Epoch 887/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6293 - val_loss: 0.4970
    Epoch 888/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6287 - val_loss: 0.4965
    Epoch 889/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6281 - val_loss: 0.4959
    Epoch 890/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6275 - val_loss: 0.4954
    Epoch 891/1000
    149/149 [==============================] - 0s 3ms/step - loss: 0.6270 - val_loss: 0.4949
    Epoch 892/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6264 - val_loss: 0.4943
    Epoch 893/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6258 - val_loss: 0.4937
    Epoch 894/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6252 - val_loss: 0.4932
    Epoch 895/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6246 - val_loss: 0.4927
    Epoch 896/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6240 - val_loss: 0.4921
    Epoch 897/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6234 - val_loss: 0.4916
    Epoch 898/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6229 - val_loss: 0.4910
    Epoch 899/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6223 - val_loss: 0.4905
    Epoch 900/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6217 - val_loss: 0.4900
    Epoch 901/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6212 - val_loss: 0.4895
    Epoch 902/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6206 - val_loss: 0.4889
    Epoch 903/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6200 - val_loss: 0.4884
    Epoch 904/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6194 - val_loss: 0.4878
    Epoch 905/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6189 - val_loss: 0.4873
    Epoch 906/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6183 - val_loss: 0.4868
    Epoch 907/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6177 - val_loss: 0.4862
    Epoch 908/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6171 - val_loss: 0.4857
    Epoch 909/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6166 - val_loss: 0.4851
    Epoch 910/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6160 - val_loss: 0.4846
    Epoch 911/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6154 - val_loss: 0.4841
    Epoch 912/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6149 - val_loss: 0.4836
    Epoch 913/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6143 - val_loss: 0.4831
    Epoch 914/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6138 - val_loss: 0.4825
    Epoch 915/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6132 - val_loss: 0.4820
    Epoch 916/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6127 - val_loss: 0.4815
    Epoch 917/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6121 - val_loss: 0.4810
    Epoch 918/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6115 - val_loss: 0.4805
    Epoch 919/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6110 - val_loss: 0.4800
    Epoch 920/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6104 - val_loss: 0.4794
    Epoch 921/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6099 - val_loss: 0.4789
    Epoch 922/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6093 - val_loss: 0.4784
    Epoch 923/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6088 - val_loss: 0.4780
    Epoch 924/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6083 - val_loss: 0.4775
    Epoch 925/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6077 - val_loss: 0.4770
    Epoch 926/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6072 - val_loss: 0.4765
    Epoch 927/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6066 - val_loss: 0.4760
    Epoch 928/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6061 - val_loss: 0.4755
    Epoch 929/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6056 - val_loss: 0.4750
    Epoch 930/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6050 - val_loss: 0.4745
    Epoch 931/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6045 - val_loss: 0.4740
    Epoch 932/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6039 - val_loss: 0.4735
    Epoch 933/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6034 - val_loss: 0.4730
    Epoch 934/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6029 - val_loss: 0.4725
    Epoch 935/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6023 - val_loss: 0.4720
    Epoch 936/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6018 - val_loss: 0.4716
    Epoch 937/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6013 - val_loss: 0.4711
    Epoch 938/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6008 - val_loss: 0.4706
    Epoch 939/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.6002 - val_loss: 0.4701
    Epoch 940/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5997 - val_loss: 0.4696
    Epoch 941/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5992 - val_loss: 0.4691
    Epoch 942/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5986 - val_loss: 0.4686
    Epoch 943/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5981 - val_loss: 0.4681
    Epoch 944/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5976 - val_loss: 0.4677
    Epoch 945/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5970 - val_loss: 0.4672
    Epoch 946/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5965 - val_loss: 0.4667
    Epoch 947/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5960 - val_loss: 0.4662
    Epoch 948/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5955 - val_loss: 0.4658
    Epoch 949/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5949 - val_loss: 0.4653
    Epoch 950/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5944 - val_loss: 0.4648
    Epoch 951/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5939 - val_loss: 0.4643
    Epoch 952/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5934 - val_loss: 0.4639
    Epoch 953/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5929 - val_loss: 0.4634
    Epoch 954/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5923 - val_loss: 0.4630
    Epoch 955/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5918 - val_loss: 0.4625
    Epoch 956/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5913 - val_loss: 0.4620
    Epoch 957/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5908 - val_loss: 0.4616
    Epoch 958/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5903 - val_loss: 0.4611
    Epoch 959/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.5898 - val_loss: 0.4607
    Epoch 960/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.5893 - val_loss: 0.4602
    Epoch 961/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.5888 - val_loss: 0.4597
    Epoch 962/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.5883 - val_loss: 0.4593
    Epoch 963/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.5878 - val_loss: 0.4588
    Epoch 964/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.5872 - val_loss: 0.4584
    Epoch 965/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5867 - val_loss: 0.4579
    Epoch 966/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5862 - val_loss: 0.4574
    Epoch 967/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5857 - val_loss: 0.4570
    Epoch 968/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5852 - val_loss: 0.4565
    Epoch 969/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5847 - val_loss: 0.4561
    Epoch 970/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5842 - val_loss: 0.4557
    Epoch 971/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5837 - val_loss: 0.4552
    Epoch 972/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5832 - val_loss: 0.4548
    Epoch 973/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5827 - val_loss: 0.4543
    Epoch 974/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5822 - val_loss: 0.4539
    Epoch 975/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5818 - val_loss: 0.4535
    Epoch 976/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5813 - val_loss: 0.4530
    Epoch 977/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5808 - val_loss: 0.4526
    Epoch 978/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5803 - val_loss: 0.4522
    Epoch 979/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5798 - val_loss: 0.4517
    Epoch 980/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5793 - val_loss: 0.4513
    Epoch 981/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5788 - val_loss: 0.4509
    Epoch 982/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5783 - val_loss: 0.4504
    Epoch 983/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5779 - val_loss: 0.4500
    Epoch 984/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5774 - val_loss: 0.4496
    Epoch 985/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5769 - val_loss: 0.4491
    Epoch 986/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5764 - val_loss: 0.4487
    Epoch 987/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5759 - val_loss: 0.4483
    Epoch 988/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.5754 - val_loss: 0.4478
    Epoch 989/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.5749 - val_loss: 0.4474
    Epoch 990/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5745 - val_loss: 0.4470
    Epoch 991/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5740 - val_loss: 0.4466
    Epoch 992/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5735 - val_loss: 0.4461
    Epoch 993/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5730 - val_loss: 0.4457
    Epoch 994/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5725 - val_loss: 0.4453
    Epoch 995/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5721 - val_loss: 0.4449
    Epoch 996/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5716 - val_loss: 0.4445
    Epoch 997/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5711 - val_loss: 0.4441
    Epoch 998/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5707 - val_loss: 0.4437
    Epoch 999/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5702 - val_loss: 0.4432
    Epoch 1000/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.5697 - val_loss: 0.4428
    

## 6.2: Plot Training and Validation Loss

Let's use the `plot_loss` helper function to take a look training and validation loss.


```python
plot_loss(history)
```


![png](output_29_0.png)


# Task 7: Predictions

## 7.1: Plot Raw Predictions

Let's use the `compare_predictions` helper function to compare predictions from the model when it was untrained and when it was trained.


```python
preds_on_trained = model.predict(X_test)

compare_predictions(preds_on_untrained, preds_on_trained, y_test)
```


![png](output_32_0.png)


## 7.2: Plot Price Predictions

The plot for price predictions and raw predictions will look the same with just one difference: The x and y axis scale is changed.


```python
price_on_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_on_trained = [convert_label_value(y) for y in preds_on_trained]
price_y_test = [convert_label_value(y) for y in y_test]

compare_predictions(price_on_untrained, price_on_trained, price_y_test)
```


![png](output_34_0.png)



```python

```
