# Notebooks – AI Engineer Associate Certificate Course

This folder contains Jupyter notebooks created while following the **AI Engineer Associate Certificate Course** (School of AI / AI Academy).

Notebooks are organized by course day, section, or project. Each subfolder includes:
- The main `.ipynb` file(s)
- A project-specific `README.md` with key observations, screenshots, and learnings
- An `images/` folder for plots, data summaries, etc.

We update this index as new work is added.

## Progress & Completed Work

| # | Section / Project                              | Status    | Key Topics Covered                                                                 | Link to Folder / Notebook                                                                 | Date Added    | Notes / Highlights                                      |
|---|------------------------------------------------|-----------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------|---------------------------------------------------------|
| 1 | Titanic – Initial EDA (Day 1 Exploration)     | Completed | Data loading from GitHub, `.info()`, `.head()`, categorical vs numerical features, value counts, `.describe()` | [titanic-eda-initial-exploration](./titanic-eda-initial-exploration/README.md)           | Jan 2026     | First project – foundational data understanding         |
| 2 | Scaling & Normalization (Iris k-NN)           | Completed | MinMaxScaler, StandardScaler, effect on distance-based model (k-NN), accuracy comparison, feature distribution visuals | [min-max-normalization](./min-max-normalization/README.md)                               | Jan 2026     | Shows scaling impact (or lack thereof) on Iris + boxplots |
| 3 | Categorical Encoding (Titanic LogReg)  | Completed | One-Hot Encoding, Label Encoding, Frequency Encoding, logistic regression baseline, impact of encoding on model performance | [encoding-categoricals](./one-hot-and-label-encoding/README.md)                 | Jan 2026     | Applies multiple encoding techniques to Titanic categoricals + evaluates with LogReg |

**Last updated:** January 31, 2026

## Current Structure
```
notebooks/
├── titanic-eda-initial-exploration/           # Project 1: Titanic EDA
│   ├── titanic_eda_day1.py
│   ├── README.md
│   └── images/
├── min-max-normalization/                     # Project 2: Scaling on Iris
│   
│   ├── min-max-scaling-and-std.py             # optional script version
│   ├── README.md
│   └── images/
├── one-hot-and-label-encoding/              # Project 3: Categorical Encoding on Titanic
│  
│   ├── one-hot-and-label-encoding.py        # or your script name
│   ├── README.md
│   └── images/
└── README.md                                  ← You are here (central index)
```

We Keep going. 🚀
