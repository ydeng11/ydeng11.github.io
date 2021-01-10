---
title: Encoding in Feature Engineering
date: 2021-01-02 13:44:11
tags: feature engineering
category:
- feature engineering
- machine learning
index_img: https://sectigostore.com/blog/wp-content/uploads/2020/04/types-of-encryption-feature.jpg
---

## Encoding in Feature Engineering

Encoding is a classical and important technique in feature engineering. Recently I learned one technique - k-fold target encoding - from the winner in Tencent Ads competition 2020. It is really new to me but apparently it is a mature technique. This makes me to write an article about these encoding techniques used in feature engineering.

### Label Encoding

Label encoding is used to transform categorical variable into numerical values. Let’s start with an example shown below where we created a dataframe has two columns, Variable and Target, and Variable has 3 classes -  A, B and C.

When we are feeding this into a model, most of models cannot handle the categorical variable directly. Simply speaking, the model doesn’t recognize A, B or C and cannot be executed thereafter. To handle this problem, we could use Label Encoding to transform these categories into numerical values. **Simply speaking, we are using numeric value to represent these strings/text in the categorical variable.**

```python
import pandas as pd
example = pd.DataFrame([["A",1], ["A", 2], ["A", 3], ["B", 4], ["B", 5], ["C", 6]], columns = ["Variable", "Target"])
example
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>6</td>
    </tr>
  </tbody>
</table>


We use the LabelEncoder from sklearn to do the transformation for this Variable. The results table below show that the A, B and C are represented by values of 0, 1, and 2. Now this Variable can be fed into our models for training.

```python
from sklearn import preprocessing
LE = preprocessing.LabelEncoder()
example["Variable_Label_Encoded"] = LE.fit_transform(example["Variable"])
example
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Target</th>
      <th>Variable_Label_Encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>6</td>
      <td>2</td>
    </tr>
  </tbody>
</table>


If we print the classes transformed by the labelencoder, we can see all three classes A, B and C here.

```pyth
LE.classes_
array(['A', 'B', 'C'], dtype=object)
```

### Count Encoding

Count Encoding is another technique to replace the categorical variables. The idea is to count the occurrences of  each class in the category and replace these classes by the corresponding count. Using the same example, we can get a count for each class as below.

```python
vc = pd.DataFrame(example["Variable"].value_counts()).reset_index().rename(columns = {"Variable" : "Variable_Count", "index" : "Variable"})
vc
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Variable_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>1</td>
    </tr>
  </tbody>
</table>


Therefore, we could join this count with our example table to replace the original classes (A, B and C) with their occurrences in this data.

```python
pd.merge(example, vc, on = "Variable", how = "left")
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Target</th>
      <th>Variable_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>


### One-hot encoding

One-hot encoding is more used to transform categorical variables with a lot of classes. We already know from above that the label encoding simply use numerical values to represent these classes so these numbers doesn’t have a meaning with it. As the results show, we have three values but their magnitude doesn’t mean anything to us. If we have many classes, it will actually make this model difficult to explain.

This is when we need one-hot encoding which could transform categorical variables into $n\ classes - 1$ binary variables.

We can use *get_dummies* function to do one-hot encoding easily. The results show that two additional columns are created to represent class B and class C respectively. If both columns are 0, then we know it must be class A.

```python
pd.get_dummies(example, "Variable", drop_first = True)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>Variable_B</th>
      <th>Variable_C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>


###  Ordinal Encoding

Ordinal Encoding is an advanced version of label encoder. Unlike label encoder only transform one categorical variable into ordinal variable, ordinal encoding could transform several categorical variables into ordinal variable simultaneously.

```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)
OrdinalEncoder()
enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
enc.transform([['Female', 3], ['Male', 1]])
array([[0., 2.],
       [1., 0.]])
```

### K-fold target encoding

K-fold target encoding originates from target encoding which replaces the categorical variables with the average target values.

```python
example.groupby("Variable")["Target"].mean()
Variable
A    2.0
B    4.5
C    6.0
Name: Target, dtype: float64
```

```python
te = pd.DataFrame(example.groupby("Variable")["Target"].mean()).reset_index().rename(columns = {"Target" : "Target_Encoding"})
pd.merge(te, example, on = "Variable", how = "inner")
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Target_Encoding</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>2.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>4.5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>4.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>6.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>


From above, we encoded each class using the average target value by the corresponding classes. But the problem is **overfitting**. When training and test dataset differ from each other a lot, this mean-target encoding would gives us incorrect results. That’s why we need k-fold target encoding. Assuming we use 5 fold here, the dataset splits into 5 folds, and we encoded the variable in one of them using the average target value from the rest 4 folds.

```python
example = pd.DataFrame([["A",1], ["A", 2], ["A", 3], ["B", 4], ["B", 5], ["C", 6],
                       ["A",2], ["A", 8], ["A", 8], ["B", 1], ["B", 2], ["C", 5],
                       ["A",3], ["A", 4], ["A", 5], ["B", 5], ["B", 4], ["C", 7],
                       ["B", 8], ["B", 7], ["C", 5],], columns = ["Variable", "Target"])
example
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>B</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>C</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>A</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>A</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>A</td>
      <td>5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>B</td>
      <td>5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>B</td>
      <td>4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>C</td>
      <td>7</td>
    </tr>
    <tr>
      <th>18</th>
      <td>B</td>
      <td>8</td>
    </tr>
    <tr>
      <th>19</th>
      <td>B</td>
      <td>7</td>
    </tr>
    <tr>
      <th>20</th>
      <td>C</td>
      <td>5</td>
    </tr>
  </tbody>
</table>


```python
idx = np.array(example.index)
np.random.shuffle(idx)
idx
array([18,  8, 13, 20, 19,  3,  7,  1, 17, 11,  2,  6, 10, 15,  4,  0,  5,
       16,  9, 12, 14], dtype=int64)
```

```python
i = 0
folds = []
while i < example.shape[0]:
    selected_fold = example.iloc[idx[i : i + 4]]
    rest_fold = example[~example.index.isin(fold1.index)]
    te = rest_fold.groupby("Variable")["Target"].mean().reset_index().rename(columns = {"Target" : "5_fold_Target_Encoding"})
    selected_fold = pd.merge(selected_fold, te, on = "Variable", how = "left").fillna(0)
    folds.append(selected_fold)
    i += 4
te_example = pd.concat(folds, axis = 0)
te_example
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Target</th>
      <th>5_fold_Target_Encoding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>7</td>
      <td>5.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>5</td>
      <td>3.875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>6</td>
      <td>5.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C</td>
      <td>5</td>
      <td>5.000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>4</td>
      <td>4.500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>2</td>
      <td>3.875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>3</td>
      <td>3.875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>1</td>
      <td>3.875</td>
    </tr>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>7</td>
      <td>4.500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
      <td>4.500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>4</td>
      <td>4.500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>5</td>
      <td>4.500</td>
    </tr>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>8</td>
      <td>4.500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>3</td>
      <td>3.875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>4</td>
      <td>3.875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>2</td>
      <td>3.875</td>
    </tr>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>2</td>
      <td>4.500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>5</td>
      <td>5.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>8</td>
      <td>3.875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>8</td>
      <td>3.875</td>
    </tr>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>5</td>
      <td>4.500</td>
    </tr>
  </tbody>
</table>


```python
example.groupby("Variable")["Target"].mean()
Variable
A    4.00
B    4.50
C    5.75
 
te_example.groupby("Variable")["5_fold_Target_Encoding"].mean()
Variable
A    3.875
B    4.500
C    5.000
```

After we did 5 fold target encoding, the value decreases a bit which helps prevent overfitting problem. **I think 5 fold target encoding makes the encoding more conservative and we won’t make aggressive estimation.**

**But I think another important application of k-fold target encoding is k-fold statistics encoding which is widely used in ML. You can see it almost everywhere in the competitions.**

The idea is the same, the dataset splits into k folds, we get the statistics for the selected fold from the rest folds. It is useful when you have a lot of numeric variables where you can pull the k-fold mean, std, skew or kurtosis. In the above example, you can also get the std from the rest 4 folds for the selected fold and insert them into the table for the following training.