# Task 1: Introduction

---

For this project, we are going to work on evaluating price of houses given the following features:

1. Year of sale of the house
2. The age of the house at the time of sale
3. Distance from city center
4. Number of stores in the locality
5. The latitude
6. The longitude


Note: This notebook uses `python 3` and these packages: `tensorflow`, `pandas`, `matplotlib`, `scikit-learn`.


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
    

# 1. Importing the Data

## 1.1: Importing the Data


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



## 1.2: Check Missing Data


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



# 2: Data Normalization

## 2.1: Data Normalization


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



## 2.2: Convert Label Value


```python
y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)

## check the first item price
print(convert_label_value(0.350088))
```

    14263
    

# 3: Create Training and Test Sets

## 3.1: Select Features


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



## 3.2: Select Labels


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



## 3.3: Feature and Label Values

We will need to extract just the numeric values for the features and labels as the TensorFlow model will expect just numeric values as input.


```python
X_arr = X.values
Y_arr = Y.values

print('X_arr shape: ', X_arr.shape)
print('Y_arr shape: ', Y_arr.shape)
```

    X_arr shape:  (5000, 6)
    Y_arr shape:  (5000,)
    

## 3.4: Train and Test Split


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
    

# 4: Create the Model

## 4.1: Create the Model

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
        optimizer='adam'
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
    

# 5: Model Training

## 5.1: Model Training

We can use an `EarlyStopping` callback from Keras to stop the model training if the validation loss stops decreasing for a few epochs.


```python
early_stopping = EarlyStopping(monitor='val_loss', patience = 5)
```

Record the random predicted value


```python
preds_on_untrained = model.predict(X_test)
```


```python
history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 1000,
    callbacks = [early_stopping]
)
```

    Epoch 1/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.6937 - val_loss: 0.3770
    Epoch 2/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.2768 - val_loss: 0.2295
    Epoch 3/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.2004 - val_loss: 0.1956
    Epoch 4/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1815 - val_loss: 0.1704
    Epoch 5/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1715 - val_loss: 0.1643
    Epoch 6/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1668 - val_loss: 0.1571
    Epoch 7/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1636 - val_loss: 0.1557
    Epoch 8/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1615 - val_loss: 0.1542
    Epoch 9/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1597 - val_loss: 0.1529
    Epoch 10/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1585 - val_loss: 0.1490
    Epoch 11/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1580 - val_loss: 0.1517
    Epoch 12/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1570 - val_loss: 0.1503
    Epoch 13/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1563 - val_loss: 0.1496
    Epoch 14/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1552 - val_loss: 0.1494
    Epoch 15/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1556 - val_loss: 0.1482
    Epoch 16/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.1543 - val_loss: 0.1480
    Epoch 17/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.1539 - val_loss: 0.1502
    Epoch 18/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1542 - val_loss: 0.1510
    Epoch 19/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1534 - val_loss: 0.1490
    Epoch 20/1000
    149/149 [==============================] - 0s 1ms/step - loss: 0.1540 - val_loss: 0.1495
    Epoch 21/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1525 - val_loss: 0.1467
    Epoch 22/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1529 - val_loss: 0.1481
    Epoch 23/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1519 - val_loss: 0.1466
    Epoch 24/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1521 - val_loss: 0.1513
    Epoch 25/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1517 - val_loss: 0.1503
    Epoch 26/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1522 - val_loss: 0.1507
    Epoch 27/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1513 - val_loss: 0.1528
    Epoch 28/1000
    149/149 [==============================] - 0s 2ms/step - loss: 0.1525 - val_loss: 0.1488
    

## 5.2: Plot Training and Validation Loss

Use the `plot_loss` helper function to take a look training and validation loss.


```python
plot_loss(history)
```


![png](output_31_0.png)


# 6: Predictions

## 6.1: Plot Raw Predictions

Use the `compare_predictions` helper function to compare predictions from the model when it was untrained and when it was trained.


```python
preds_on_trained = model.predict(X_test)

compare_predictions(preds_on_untrained, preds_on_trained, y_test)
```


![png](output_34_0.png)


## 6.2: Plot Price Predictions

The plot for price predictions and raw predictions will look the same with just one difference: The x and y axis scale is changed.


```python
price_on_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_on_trained = [convert_label_value(y) for y in preds_on_trained]
price_y_test = [convert_label_value(y) for y in y_test]

compare_predictions(price_on_untrained, price_on_trained, price_y_test)
```


![png](output_36_0.png)



```python

```
