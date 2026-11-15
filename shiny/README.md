# Marriage and Affairs — Shiny app

Interactive look at the Affairs data and an R refit of the two models from `final_portfolio/final/final_portfolio.py`: logistic regression and a depth-3 classification tree. Occupation and husband's occupation can be charted, and they are excluded from both models.

This app does not use the portfolio site's layout.

## Run

From this directory:

```bash
cd /Users/juancortes/Projects/Marriage-and-Affairs/shiny
Rscript -e 'shiny::runApp(".", host = "127.0.0.1", port = 7735, launch.browser = FALSE)'
```

Then open `http://127.0.0.1:7735`.

`shiny::runApp()` uses this folder as the working directory. The data file is read from `../final_portfolio/final/_data/affairs.csv`, relative to this folder.

## Packages

`shiny`, `bslib`, `ggplot2`, `dplyr`, `rpart`, `rpart.plot`, `scales`.

If one is missing, install it into your user library (no sudo):

```r
install.packages(c("shiny", "bslib", "ggplot2", "dplyr", "rpart", "rpart.plot", "scales"))
```

## What you can change

On **Explore**, pick a predictor and switch the level summary between the affair rate (`affairs > 0`) and the mean of the original affairs measure. The distribution, the by-level chart, the sentence, and the table update. A separate control shows the class balance of `had_affair`, the original affairs measure, or both.

On **Models**, the logistic coefficients (odds ratios and Wald intervals) and the tree splits are the R refit. Python metrics, when shown, are labeled as scikit-learn and are not the R numbers.

On **Why and what next**, logistic regression, ordinary linear regression, and the shallow tree are explained in short. Poisson or negative binomial, regularized logistic, and a tree ensemble are listed as not fit, with no scores.
