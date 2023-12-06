# FNC-1 dataset

The data provided is `(headline, body, stance)` instances, where `stance` is one of `{unrelated, discuss, agree, disagree}`. The dataset is provided as two CSVs:


### `train_bodies.csv`

This file contains the body text of articles (the `articleBody` column) with corresponding IDs (`Body ID`)

### `train_stances.csv`

This file contains the labeled stances (the `Stance` column) for pairs of article headlines (`Headline`) and article bodies (`Body ID`, referring to entries in `train_bodies.csv`).

### Distribution of the data

The distribution of `Stance` classes in `train_stances.csv` is as follows:

|   rows |   unrelated |   discuss |     agree |   disagree |
|-------:|------------:|----------:|----------:|-----------:|
|  49972 |    0.73131  |  0.17828  | 0.0736012 |  0.0168094 |

### Split of the data

As the official only supply us a training set, we will split it into train/dev/test in the ratio of 8:1:1 for our project.
As `train.csv`, `val.csv`, `test.csv`.

# LIAR dataset

The size of the LIAR dataset in train/dev/test are 10240/1284/1284. The dataset is mainly consist of Statements and Labels. The dataset consists of statements or claims made by various individuals. Each statement is labeled with one of six possible labels indicating the degree of truthfulness: False, Barely True, Half True, Mostly True, True, Pants on Fire (for egregiously false statements).

### Description of the TSV format:

Column 1: the ID of the statement ([ID].json).
Column 2: the label.
Column 3: the statement.
Column 4: the subject(s).
Column 5: the speaker.
Column 6: the speaker's job title.
Column 7: the state info.
Column 8: the party affiliation.
Column 9-13: the total credit history count, including the current statement.
9: barely true counts.
10: false counts.
11: half true counts.
12: mostly true counts.
13: pants on fire counts.
Column 14: the context (venue / location of the speech or statement).
