# Marriage and Affairs — interactive Shiny app
#
# How to run (from this shiny/ directory):
#   Rscript -e 'shiny::runApp(".", host = "127.0.0.1", port = 7735, launch.browser = FALSE)'
#
# From any working directory:
#   shiny::runApp("/absolute/path/to/Marriage-and-Affairs/shiny")
#
# shiny::runApp() sets the working directory to this folder before sourcing
# app.R. The CSV is resolved from this file's app directory:
#   ../final_portfolio/final/_data/affairs.csv
# Do not launch it in a way that expects the repository root to be the
# working directory.
#
# Packages: shiny, bslib, ggplot2, dplyr, rpart, rpart.plot, scales.
# The logistic model is an unpenalized binomial glm. The tree is rpart with
# Gini splits and maxdepth 3. Metrics below are this R refit.

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(ggplot2)
  library(dplyr)
  library(rpart)
  library(rpart.plot)
  library(scales)
})

find_app_dir <- function() {
  # Prefer the path of this file. shiny::runApp() sources app.R and sets
  # ofile; Rscript app.R passes --file=. getwd() is the fallback, and runApp
  # sets that to the app directory before sourcing.
  for (i in seq_len(sys.nframe())) {
    env <- sys.frame(i)
    if (exists("ofile", envir = env, inherits = FALSE)) {
      ofile <- get("ofile", envir = env, inherits = FALSE)
      if (is.character(ofile) && length(ofile) == 1L && nzchar(ofile)) {
        return(dirname(normalizePath(ofile, mustWork = FALSE)))
      }
    }
  }
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = FALSE)))
  }
  getwd()
}

app_dir <- find_app_dir()
csv_path <- file.path(app_dir, "..", "final_portfolio", "final", "_data", "affairs.csv")
if (!file.exists(csv_path)) {
  stop(
    "Could not find affairs.csv at ", csv_path,
    ". Launch with shiny::runApp() so the app directory is the shiny folder.",
    call. = FALSE
  )
}
csv_path <- normalizePath(csv_path)

raw <- utils::read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE)
needed <- c(
  "rate_marriage", "age", "yrs_married", "children", "religious",
  "educ", "occupation", "occupation_husb", "affairs"
)
missing_cols <- setdiff(needed, names(raw))
if (length(missing_cols)) {
  stop("affairs.csv is missing: ", paste(missing_cols, collapse = ", "), call. = FALSE)
}

affairs <- raw[, needed]
if (anyNA(affairs)) {
  stop("affairs.csv has missing values in the analysis columns.", call. = FALSE)
}
affairs$had_affair <- as.integer(affairs$affairs > 0)

fmt_num <- function(x) {
  x <- as.numeric(x)
  vapply(x, function(v) {
    if (!is.finite(v)) {
      return(NA_character_)
    }
    if (abs(v - round(v)) < 1e-6) {
      format(round(v), trim = TRUE, scientific = FALSE)
    } else {
      sub("\\.$", "", sub("0+$", "", sprintf("%.4f", v)))
    }
  }, character(1))
}

fmt_n <- function(x) format(as.integer(round(x)), big.mark = ",", trim = TRUE)
fmt_3 <- function(x) sprintf("%.3f", x)
fmt_pct1 <- function(x) sprintf("%.1f%%", 100 * x)
fmt_pct2 <- function(x) sprintf("%.2f%%", 100 * x)

fmt_p <- function(p) {
  ifelse(is.na(p), "—", ifelse(p < 0.001, "< 0.001", sprintf("%.3f", p)))
}

join_or <- function(x) {
  x <- as.character(x)
  n <- length(x)
  if (n == 0) {
    return("none")
  }
  if (n == 1) {
    return(x)
  }
  if (n == 2) {
    return(paste(x, collapse = " or "))
  }
  paste0(paste(x[-n], collapse = ", "), ", or ", x[n])
}

var_titles <- c(
  rate_marriage = "marriage rating",
  age = "age",
  yrs_married = "years married",
  children = "children",
  religious = "religious commitment",
  educ = "education",
  occupation = "respondent occupation",
  occupation_husb = "husband occupation"
)

codebook <- list(
  rate_marriage = list(
    title = "Marriage rating",
    in_model = TRUE,
    blurb = "Self-rated marital satisfaction. The report codes 1 as very poor and 5 as very good. The fitted models use this number as a single slope.",
    labels = c(
      "1" = "1  Very poor",
      "2" = "2",
      "3" = "3",
      "4" = "4",
      "5" = "5  Very good"
    )
  ),
  age = list(
    title = "Age",
    in_model = TRUE,
    blurb = "Age is stored as six codes: 17.5, 22, 27, 32, 37, and 42. The fitted models use those numbers.",
    labels = NULL
  ),
  yrs_married = list(
    title = "Years married",
    in_model = TRUE,
    blurb = "Years married is stored as 0.5, 2.5, 6, 9, 13, 16.5, and 23. The fitted models use those numbers.",
    labels = NULL
  ),
  children = list(
    title = "Children",
    in_model = TRUE,
    blurb = "Number of children. The highest code stored in the file is 5.5. The fitted models use the number itself.",
    labels = NULL
  ),
  religious = list(
    title = "Religious commitment",
    in_model = TRUE,
    blurb = "Religious commitment is 1, 2, 3, or 4 in this file. The report codes 1 as not religious and 4 as strongly religious. It is not a 1–5 scale.",
    labels = c(
      "1" = "1  Not religious",
      "2" = "2",
      "3" = "3",
      "4" = "4  Strongly religious"
    )
  ),
  educ = list(
    title = "Education",
    in_model = TRUE,
    blurb = "Education codes from the report: 9 grade school, 12 high school, 14 some college, 16 college graduate, 17 some graduate school, 20 advanced degree. The fitted models use one slope on this code.",
    labels = c(
      "9" = "9  Grade school",
      "12" = "12  High school",
      "14" = "14  Some college",
      "16" = "16  College graduate",
      "17" = "17  Some graduate school",
      "20" = "20  Advanced degree"
    )
  ),
  occupation = list(
    title = "Respondent occupation",
    in_model = FALSE,
    blurb = "Respondent occupation, coded 1–6 in the analysis script: 1 student, 2 agriculture, 3 white collar, 4 teacher/nurse, 5 managerial, 6 professional.",
    labels = c(
      "1" = "1  Student",
      "2" = "2  Agriculture",
      "3" = "3  White collar",
      "4" = "4  Teacher / nurse",
      "5" = "5  Managerial",
      "6" = "6  Professional"
    )
  ),
  occupation_husb = list(
    title = "Husband occupation",
    in_model = FALSE,
    blurb = "Husband's occupation, same 1–6 codes as the respondent: 1 student, 2 agriculture, 3 white collar, 4 teacher/nurse, 5 managerial, 6 professional.",
    labels = c(
      "1" = "1  Student",
      "2" = "2  Agriculture",
      "3" = "3  White collar",
      "4" = "4  Teacher / nurse",
      "5" = "5  Managerial",
      "6" = "6  Professional"
    )
  )
)

label_for <- function(var, value) {
  key <- fmt_num(value)
  labs <- codebook[[var]]$labels
  if (!is.null(labs) && key %in% names(labs)) labs[[key]] else key
}

predictor_choices <- c(
  "Marriage rating" = "rate_marriage",
  "Age" = "age",
  "Years married" = "yrs_married",
  "Children" = "children",
  "Religious commitment" = "religious",
  "Education" = "educ",
  "Occupation (exploratory)" = "occupation",
  "Husband occupation (exploratory)" = "occupation_husb"
)

model_vars <- c("rate_marriage", "age", "yrs_married", "children", "religious", "educ")

stratified_split <- function(y, test_frac = 0.2, seed = 42) {
  RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  set.seed(seed)
  y <- as.integer(y)
  test_idx <- integer(0)
  for (cls in sort(unique(y))) {
    idx <- which(y == cls)
    n_test <- round(length(idx) * test_frac)
    test_idx <- c(test_idx, sample(idx, n_test))
  }
  test_idx <- sort(test_idx)
  list(train = setdiff(seq_along(y), test_idx), test = test_idx)
}

class_report <- function(y, pred) {
  y <- as.integer(y)
  pred <- as.integer(pred)
  one <- function(cls, name) {
    tp <- sum(pred == cls & y == cls)
    fp <- sum(pred == cls & y != cls)
    fn <- sum(pred != cls & y == cls)
    precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
    f1 <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
      NA_real_
    } else {
      2 * precision * recall / (precision + recall)
    }
    data.frame(
      Class = name,
      Precision = fmt_3(precision),
      Recall = fmt_3(recall),
      F1 = fmt_3(f1),
      Support = fmt_n(sum(y == cls)),
      stringsAsFactors = FALSE
    )
  }
  rbind(one(0L, "No affair"), one(1L, "Affair"))
}

confusion_table <- function(y, pred) {
  y <- as.integer(y)
  pred <- as.integer(pred)
  data.frame(
    Actual = c("No affair", "Affair"),
    `Predicted no affair` = fmt_n(c(sum(y == 0 & pred == 0), sum(y == 1 & pred == 0))),
    `Predicted affair` = fmt_n(c(sum(y == 0 & pred == 1), sum(y == 1 & pred == 1))),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

parse_split <- function(text) {
  hit <- regmatches(
    text,
    regexec("^([A-Za-z_]+)\\s*(<=|>=|<|>)\\s*([-+0-9.]+)$", text)
  )[[1]]
  if (length(hit) != 4) {
    return(NULL)
  }
  list(var = hit[2], op = hit[3], cut = as.numeric(hit[4]))
}

leaf_rules <- function(fit, data) {
  frame <- fit$frame
  leaf_ids <- rownames(frame)[frame$var == "<leaf>"]
  paths <- path.rpart(fit, nodes = as.integer(leaf_ids), print.it = FALSE)
  rows <- lapply(leaf_ids, function(id) {
    steps <- paths[[id]]
    if (is.null(steps)) {
      stop("rpart path missing for node ", id, call. = FALSE)
    }
    steps <- steps[steps != "root"]
    parsed <- lapply(steps, parse_split)
    parsed <- parsed[!vapply(parsed, is.null, logical(1))]
    pieces <- character(0)
    for (v in model_vars) {
      sps <- parsed[vapply(parsed, function(p) identical(p$var, v), logical(1))]
      if (!length(sps)) {
        next
      }
      vals <- sort(unique(data[[v]]))
      for (sp in sps) {
        if (sp$op == "<") {
          vals <- vals[vals < sp$cut - 1e-8]
        } else if (sp$op == "<=") {
          vals <- vals[vals <= sp$cut + 1e-8]
        } else if (sp$op == ">") {
          vals <- vals[vals > sp$cut + 1e-8]
        } else if (sp$op == ">=") {
          vals <- vals[vals >= sp$cut - 1e-8]
        }
      }
      shown <- if (length(vals)) join_or(fmt_num(vals)) else "no recorded code"
      pieces <- c(pieces, sprintf("%s is %s", var_titles[[v]], shown))
    }
    rule <- if (length(pieces)) paste(pieces, collapse = "; ") else "all training rows"
    idx <- match(id, rownames(frame))
    yv <- as.integer(frame$yval[idx])
    data.frame(
      Predicted = if (yv == 2L) "Affair" else "No affair",
      `Training rows` = fmt_n(frame$n[idx]),
      `Affair share` = sprintf("%.1f%%", 100 * frame$yval2[idx, 5]),
      Rule = rule,
      share_num = frame$yval2[idx, 5],
      yv = yv,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  })
  out <- do.call(rbind, rows)
  out <- out[order(out$yv, -out$share_num), , drop = FALSE]
  out$share_num <- NULL
  out$yv <- NULL
  rownames(out) <- NULL
  out
}

leaf_depth <- function(id) {
  d <- 0L
  n <- as.integer(id)
  while (n > 1L) {
    n <- n %/% 2L
    d <- d + 1L
  }
  d
}

split_story <- function(fit) {
  frame <- fit$frame
  root_var <- as.character(frame$var[rownames(frame) == "1"])
  next_ids <- intersect(c("2", "3"), rownames(frame))
  next_vars <- as.character(frame[next_ids, "var"])
  next_vars <- next_vars[next_vars != "<leaf>"]
  root_phrase <- if (root_var %in% names(var_titles)) var_titles[[root_var]] else root_var
  if (length(next_vars) && all(next_vars == "yrs_married")) {
    next_phrase <- "Both of those branches then split on years married."
  } else if (length(next_vars)) {
    nice <- var_titles[unique(next_vars)]
    nice[is.na(nice)] <- unique(next_vars)[is.na(nice)]
    next_phrase <- sprintf("The next splits use %s.", join_or(nice))
  } else {
    next_phrase <- "Both children of the root are leaves."
  }
  leaf_ids <- rownames(frame)[frame$var == "<leaf>"]
  depths <- vapply(leaf_ids, leaf_depth, integer(1))
  changes <- any(vapply(leaf_ids, function(id) {
    parent <- as.character(as.integer(id) %/% 2L)
    frame[id, "yval"] != frame[parent, "yval"]
  }, logical(1)))
  depth_phrase <- if (length(depths) && all(depths == 3L) && !changes) {
    "Every leaf sits at depth 3. Those last splits do not change the predicted label."
  } else if (length(depths) && all(depths == 3L) && changes) {
    "Every leaf sits at depth 3. At least one of those last splits changes the predicted label."
  } else {
    "The leaf table lists the paths this fit actually kept."
  }
  paste(
    sprintf("In this R tree the root split is %s.", root_phrase),
    next_phrase,
    depth_phrase
  )
}

split_idx <- stratified_split(affairs$had_affair, test_frac = 0.2, seed = 42)
train <- affairs[split_idx$train, ]
test <- affairs[split_idx$test, ]
train$had_f <- factor(train$had_affair, levels = c(0, 1), labels = c("No affair", "Affair"))

logit <- glm(
  had_affair ~ rate_marriage + age + yrs_married + children + religious + educ,
  data = train,
  family = binomial(link = "logit")
)
logit_prob_test <- as.numeric(predict(logit, newdata = test, type = "response"))
logit_pred_test <- as.integer(logit_prob_test >= 0.5)
logit_prob_train <- as.numeric(predict(logit, newdata = train, type = "response"))
logit_pred_train <- as.integer(logit_prob_train >= 0.5)

tree <- rpart(
  had_f ~ rate_marriage + age + yrs_married + children + religious + educ,
  data = train,
  method = "class",
  parms = list(split = "gini"),
  control = rpart.control(
    maxdepth = 3,
    cp = -1,
    minsplit = 2,
    minbucket = 1,
    xval = 0
  )
)
tree_pred_test <- as.integer(as.character(predict(tree, newdata = test, type = "class")) == "Affair")
tree_pred_train <- as.integer(as.character(predict(tree, newdata = train, type = "class")) == "Affair")

term_labels <- c(
  "(Intercept)" = "Intercept",
  rate_marriage = "Marriage rating",
  age = "Age",
  yrs_married = "Years married",
  children = "Children",
  religious = "Religious commitment",
  educ = "Education"
)

logit_sm <- summary(logit)$coefficients
logit_ci <- exp(confint.default(logit))
logit_or <- exp(coef(logit))
coef_table <- data.frame(
  Term = unname(term_labels[rownames(logit_sm)]),
  Coefficient = fmt_3(logit_sm[, 1]),
  `Std. error` = fmt_3(logit_sm[, 2]),
  `Odds ratio` = fmt_3(logit_or),
  `95% low` = fmt_3(logit_ci[, 1]),
  `95% high` = fmt_3(logit_ci[, 2]),
  `p value` = fmt_p(logit_sm[, 4]),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
if (anyNA(coef_table$Term)) {
  coef_table$Term[is.na(coef_table$Term)] <- rownames(logit_sm)[is.na(coef_table$Term)]
}

score_pack <- function(y, pred) {
  correct <- sum(as.integer(y) == as.integer(pred))
  n <- length(y)
  acc <- correct / n
  list(
    correct = correct,
    n = n,
    acc = acc,
    acc_label = fmt_pct2(acc),
    acc_exact = sprintf("%.4f", acc),
    fraction = sprintf("%s / %s", fmt_n(correct), fmt_n(n)),
    report = class_report(y, pred),
    confusion = confusion_table(y, pred)
  )
}

n_all <- nrow(affairs)
n_none <- sum(affairs$had_affair == 0L)
n_affair <- sum(affairs$had_affair == 1L)
overall_rate <- mean(affairs$had_affair)
overall_mean <- mean(affairs$affairs)
pos <- affairs[affairs$affairs > 0, ]

fitted_models <- list(
  logit = score_pack(test$had_affair, logit_pred_test),
  logit_train = score_pack(train$had_affair, logit_pred_train),
  tree = score_pack(test$had_affair, tree_pred_test),
  tree_train = score_pack(train$had_affair, tree_pred_train),
  tree_fit = tree,
  coef_table = coef_table,
  leaf_table = leaf_rules(tree, affairs),
  split_story = split_story(tree),
  n_train = nrow(train),
  n_test = nrow(test),
  train_none = sum(train$had_affair == 0L),
  train_affair = sum(train$had_affair == 1L),
  test_none = sum(test$had_affair == 0L),
  test_affair = sum(test$had_affair == 1L)
)

message(
  "R logistic test accuracy ", fitted_models$logit$acc_exact,
  " (", fitted_models$logit$fraction, ")"
)
message(
  "R tree test accuracy ", fitted_models$tree$acc_exact,
  " (", fitted_models$tree$fraction, ")"
)
message(
  "Class balance n=", n_all, " none=", n_none, " affair=", n_affair
)

html_table <- function(df) {
  headers <- paste0(
    "<th>",
    htmltools::htmlEscape(names(df)),
    "</th>",
    collapse = ""
  )
  rows <- apply(df, 1, function(r) {
    paste0(
      "<tr>",
      paste0("<td>", htmltools::htmlEscape(as.character(r)), "</td>", collapse = ""),
      "</tr>"
    )
  })
  htmltools::HTML(paste0(
    "<div class='aff-wrap'><table class='aff-table'><thead><tr>",
    headers,
    "</tr></thead><tbody>",
    paste(rows, collapse = ""),
    "</tbody></table></div>"
  ))
}

theme_affairs <- function(compact = FALSE) {
  theme_minimal(base_size = if (compact) 11 else 13) +
    theme(
      plot.title = element_text(
        face = "bold",
        size = if (compact) 12 else 14,
        colour = "#1d2a27",
        lineheight = 1.05
      ),
      plot.subtitle = element_text(
        size = if (compact) 9.5 else 11,
        colour = "#5c675f",
        margin = margin(b = 8),
        lineheight = 1.05
      ),
      plot.title.position = "plot",
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(colour = "#ece4d8", linewidth = 0.4),
      axis.text = element_text(colour = "#1d2a27"),
      axis.title.x = element_text(colour = "#3d4a46", margin = margin(t = 8)),
      axis.title.y = element_blank(),
      plot.margin = margin(8, if (compact) 10 else 18, 6, 4)
    )
}

output_px <- function(session, output_id) {
  px <- session$clientData[[paste0("output_", output_id, "_width")]]
  if (is.null(px) || length(px) == 0 || is.na(px)) NA_real_ else as.numeric(px)
}

wrap_plot_text <- function(text, px) {
  chars <- if (is.na(px)) 40 else max(16, floor(px / 8.8))
  paste(strwrap(text, width = chars), collapse = "\n")
}

summarise_var <- function(data, var) {
  s <- data %>%
    group_by(level = .data[[var]]) %>%
    summarise(
      n = n(),
      affair_rate = mean(.data$had_affair),
      mean_affairs = mean(.data$affairs),
      .groups = "drop"
    ) %>%
    arrange(.data$level)
  s$label <- factor(
    vapply(s$level, function(v) label_for(var, v), character(1)),
    levels = vapply(s$level, function(v) label_for(var, v), character(1))
  )
  s
}

app_theme <- bs_theme(
  version = 5,
  bg = "#f3eee6",
  fg = "#1d2a27",
  primary = "#0f6e68",
  secondary = "#a34b32",
  base_font = font_collection(
    "Avenir Next", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"
  ),
  heading_font = font_collection(
    "Iowan Old Style", "Palatino Linotype", "Palatino", "Georgia", "serif"
  )
)

app_css <- "
.navbar {
  background: #f7f3ec !important;
  border-bottom: 3px solid #a34b32;
}
.navbar-brand {
  font-family: 'Iowan Old Style', Palatino, Georgia, serif;
  font-weight: 650;
}
.navbar .nav-link { color: #24312e; }
.navbar .nav-link.active {
  color: #0f6e68 !important;
  font-weight: 700;
}
.lede { max-width: 46rem; font-size: 1.05rem; }
.kicker {
  letter-spacing: 0.11em;
  text-transform: uppercase;
  font-size: 0.72rem;
  color: #7a7168;
  margin-bottom: 0.25rem;
}
.callout {
  background: #f4f7f4;
  border-left: 4px solid #0f6e68;
  padding: 0.7rem 0.9rem;
  border-radius: 0 10px 10px 0;
  margin-top: 0.35rem;
}
.callout.warn {
  background: #fbf4ef;
  border-left-color: #a34b32;
}
.callout p { margin: 0 0 0.35rem; }
.callout p:last-child { margin-bottom: 0; }
#pattern {
  font-size: 1.02rem;
  margin: 0.2rem 0 0.6rem;
}
.aff-wrap { overflow-x: auto; }
.aff-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
.aff-table th, .aff-table td {
  text-align: left;
  padding: 0.45rem 0.55rem;
  border-bottom: 1px solid #e6dfd4;
  vertical-align: top;
}
.aff-table th {
  color: #6b645c;
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.note {
  color: #4e5854;
  font-size: 0.92rem;
}
.app-foot {
  color: #6e675f;
  font-size: 0.82rem;
  padding: 0.4rem 0.2rem 0.2rem;
}
.python-card { background: #f7f3ec; }
details summary { cursor: pointer; font-weight: 650; }
"

ui <- page_navbar(
  title = "Marriage and Affairs",
  id = "mainnav",
  theme = app_theme,
  fillable = FALSE,
  window_title = "Marriage and Affairs",
  header = tags$head(tags$style(HTML(app_css))),
  footer = div(
    class = "app-foot",
    "Juan Cortes · Affairs data · Logistic regression and a depth-3 tree refit in R · Occupation charts are exploratory"
  ),
  nav_panel(
    "Explore",
    p(
      class = "lede",
      "Choose a predictor. The distribution and the affair pattern redraw for that variable. Counts below are computed from the file."
    ),
    layout_column_wrap(
      width = "13rem",
      fill = FALSE,
      value_box(
        title = "Respondents",
        value = fmt_n(n_all),
        p("Rows in affairs.csv"),
        theme = value_box_theme(bg = "#1d2a27", fg = "#f7f3ec"),
        fill = FALSE
      ),
      value_box(
        title = "No affair",
        value = fmt_n(n_none),
        p("affairs = 0, so had_affair = 0"),
        theme = value_box_theme(bg = "#0f6e68", fg = "#f4fbf9"),
        fill = FALSE
      ),
      value_box(
        title = "At least one",
        value = fmt_n(n_affair),
        p("affairs > 0, so had_affair = 1"),
        theme = value_box_theme(bg = "#a34b32", fg = "#fff8f5"),
        fill = FALSE
      ),
      value_box(
        title = "Affair rate",
        value = fmt_pct1(overall_rate),
        p("Share with had_affair = 1"),
        theme = value_box_theme(bg = "#f7f3ec", fg = "#1d2a27"),
        fill = FALSE
      )
    ),
    card(
      card_header("Predictor"),
      layout_columns(
        col_widths = breakpoints(sm = 12, md = c(5, 7)),
        selectInput(
          "var",
          "Variable",
          choices = predictor_choices,
          selected = "rate_marriage",
          selectize = FALSE,
          width = "100%"
        ),
        radioButtons(
          "metric",
          "Summarize each level by",
          choices = c(
            "Affair rate (share with affairs > 0)" = "rate",
            "Mean of the affairs measure" = "mean"
          ),
          selected = "rate",
          inline = TRUE
        )
      ),
      uiOutput("var_note")
    ),
    textOutput("pattern", container = tags$p),
    layout_columns(
      col_widths = breakpoints(sm = 12, lg = 6),
      card(
        card_header("Distribution"),
        plotOutput("dist_plot", height = "440px")
      ),
      card(
        card_header("By level"),
        plotOutput("level_plot", height = "440px")
      )
    ),
    card(
      card_header("Levels"),
      tableOutput("level_table"),
      p(
        class = "note",
        "This table uses the full file, before the train/test split. Affair rate is the share with affairs > 0. The mean is the original affairs measure, zeros included."
      )
    ),
    card(
      card_header("Outcome: binary target and the original measure"),
      radioButtons(
        "outcome_view",
        "Show",
        choices = c(
          "Class balance and the affairs measure" = "both",
          "Class balance only" = "balance",
          "Affairs measure only" = "measure"
        ),
        selected = "both",
        inline = TRUE
      ),
      p(
        class = "note",
        sprintf(
          "had_affair is 1 when affairs > 0, and 0 when affairs is 0. %s people sit at exactly 0. %s people are above 0. Among those above 0, the affairs measure has median %s and maximum %s. The binary target treats a short and a long measure as the same class.",
          fmt_n(n_none),
          fmt_n(n_affair),
          sprintf("%.2f", stats::median(pos$affairs)),
          sprintf("%.2f", max(pos$affairs))
        )
      ),
      conditionalPanel(
        condition = "input.outcome_view == 'balance' || input.outcome_view == 'both'",
        plotOutput("balance_plot", height = "260px")
      ),
      conditionalPanel(
        condition = "input.outcome_view == 'measure' || input.outcome_view == 'both'",
        plotOutput("measure_plot", height = "300px")
      )
    )
  ),
  nav_panel(
    "Models",
    p(
      class = "lede",
      "Two models were fit in R for this app: an unpenalized logistic regression and a depth-3 classification tree. Ordinary linear regression was not fit."
    ),
    layout_column_wrap(
      width = "18rem",
      fill = FALSE,
      value_box(
        title = "Logistic test accuracy",
        value = fitted_models$logit$acc_label,
        p(sprintf("R glm · %s correct", fitted_models$logit$fraction)),
        theme = value_box_theme(bg = "#0f6e68", fg = "#f4fbf9"),
        fill = FALSE
      ),
      value_box(
        title = "Tree test accuracy",
        value = fitted_models$tree$acc_label,
        p(sprintf("R rpart · %s correct", fitted_models$tree$fraction)),
        theme = value_box_theme(bg = "#1d2a27", fg = "#f7f3ec"),
        fill = FALSE
      )
    ),
    card(
      card_header("What was fit"),
      p(
        sprintf(
          "Stratified 80/20 split in R, set.seed(42). Training rows: %s (%s with no affair, %s with an affair). Test rows: %s (%s and %s). A predicted affair is a probability of at least 0.5.",
          fmt_n(fitted_models$n_train),
          fmt_n(fitted_models$train_none),
          fmt_n(fitted_models$train_affair),
          fmt_n(fitted_models$n_test),
          fmt_n(fitted_models$test_none),
          fmt_n(fitted_models$test_affair)
        )
      ),
      p(
        "Predictors: marriage rating, age, years married, children, religious commitment, and education. Occupation and husband's occupation are excluded. Each predictor enters as the numeric code in the file. The codes are not one-hot encoded, so the education coefficient is one slope across 9, 12, 14, 16, 17, and 20, and the religious coefficient is one slope across 1–4."
      ),
      p(
        "The logistic model is glm with family binomial and the logit link: unpenalized maximum likelihood. scikit-learn's LogisticRegression, which produced the Python figures below, applies an L2 penalty by default (C = 1). The R split is also a different draw than random_state=42 in scikit-learn. The accuracies are not expected to match to the last decimal."
      ),
      tags$details(
        tags$summary("Tree settings"),
        p(
          class = "note",
          "rpart method class, split = gini, maxdepth = 3, minsplit = 2, minbucket = 1, xval = 0. rpart's default complexity parameter skips a split that does not reduce the number of misclassified training rows. cp is set to -1 so a Gini improvement can be kept out to depth 3, which is the depth limit on the Python tree."
        )
      )
    ),
    layout_columns(
      col_widths = breakpoints(sm = 12, lg = 6),
      card(
        card_header("Logistic regression, test set"),
        p(sprintf(
          "Test accuracy %s (%s). Training accuracy %s on %s rows.",
          fitted_models$logit$acc_label,
          fitted_models$logit$fraction,
          fitted_models$logit_train$acc_label,
          fmt_n(fitted_models$n_train)
        )),
        html_table(fitted_models$logit$report),
        p(class = "note", "Rows are the actual class. Columns are the predicted class."),
        html_table(fitted_models$logit$confusion)
      ),
      card(
        card_header("Classification tree, test set"),
        p(sprintf(
          "Test accuracy %s (%s). Training accuracy %s on %s rows.",
          fitted_models$tree$acc_label,
          fitted_models$tree$fraction,
          fitted_models$tree_train$acc_label,
          fmt_n(fitted_models$n_train)
        )),
        html_table(fitted_models$tree$report),
        p(class = "note", "Rows are the actual class. Columns are the predicted class."),
        html_table(fitted_models$tree$confusion)
      )
    ),
    card(
      card_header("Logistic coefficients and odds ratios"),
      p(
        class = "note",
        "Odds ratios are exp(coefficient): the multiplicative change in the odds of an affair for a one-point increase in that numeric code, with the other codes held fixed. Intervals are Wald 95% intervals on the odds ratio. The intercept is the log-odds when every predictor is 0, which does not occur in this file."
      ),
      html_table(coef_table)
    ),
    card(
      card_header("Tree splits"),
      p(fitted_models$split_story),
      p(
        class = "note",
        "Rules are written in the recorded codes, not the raw numeric cut. Affair share and row counts are from the training leaf. In the drawing, each node shows the fitted class, the training probability of Affair, and the share of training rows."
      ),
      html_table(fitted_models$leaf_table),
      plotOutput("tree_plot", height = "680px")
    ),
    card(
      class = "python-card",
      card_header("scikit-learn context, separate from the R refit"),
      p(
        "A scikit-learn refit of the analysis script has been reported at logistic test accuracy 70.96%, tree test accuracy 70.17%, no-affair recall 0.875, affair recall 0.363, and affair precision 0.580. Those figures come from LogisticRegression(max_iter = 10000) and DecisionTreeClassifier(max_depth = 3, criterion = 'gini', random_state = 42). They are not the R metrics on this page."
      )
    )
  ),
  nav_panel(
    "Why and what next",
    layout_columns(
      col_widths = breakpoints(sm = 12, lg = 4),
      card(
        card_header("Why logistic regression"),
        p("had_affair is a yes/no outcome. Logistic regression models the log-odds of a 1, so the fitted probability stays between 0 and 1."),
        p("The predictors enter as the numeric codes in the file. That is the specification in the analysis script. This app does not fit a one-hot version of those codes.")
      ),
      card(
        card_header("Why ordinary linear regression is a poor fit"),
        p("Ordinary least squares on a 0/1 target can predict values below 0 or above 1, and it treats the outcome as a continuous score. It was not the model that was fit."),
        p("The fitted regression is the binomial logit model on the Models tab.")
      ),
      card(
        card_header("Why a shallow tree"),
        p("A depth-3 classification tree is there to be read: a few Gini splits and a label in each leaf. On this R refit the readable cuts are marriage rating, then years married."),
        p(fitted_models$split_story)
      )
    ),
    card(
      card_header("Other models — not fit"),
      p(
        "Binarizing the affairs measure throws away duration. Someone with a trace and someone with a long measure both become had_affair = 1. Models that keep the original measure, or that regularize or ensemble the binary model, were not fit. No scores are shown for them."
      ),
      html_table(data.frame(
        Approach = c(
          "Poisson or negative binomial",
          "Regularized logistic regression",
          "Tree ensemble"
        ),
        `Would use` = c(
          "The original affairs measure, so duration is kept",
          "The same binary target, with a penalty on the coefficients",
          "The same binary target, with many trees instead of one depth-3 tree"
        ),
        Status = c("Not fit", "Not fit", "Not fit"),
        check.names = FALSE,
        stringsAsFactors = FALSE
      ))
    )
  )
)

server <- function(input, output, session) {
  var_summary <- reactive({
    req(input$var)
    summarise_var(affairs, input$var)
  })

  output$var_note <- renderUI({
    meta <- codebook[[input$var]]
    tags$div(
      class = if (isTRUE(meta$in_model)) "callout" else "callout warn",
      tags$p(meta$blurb),
      if (!isTRUE(meta$in_model)) {
        tags$p(
          tags$strong("Exploratory only. "),
          "The logistic regression and the tree both exclude occupation and occupation_husb. This chart is not a model effect."
        )
      }
    )
  })

  output$pattern <- renderText({
    s <- var_summary()
    meta <- codebook[[input$var]]
    if (identical(input$metric, "mean")) {
      i_lo <- which.min(s$mean_affairs)
      i_hi <- which.max(s$mean_affairs)
      sprintf(
        "Mean affairs measure by %s runs from %s at %s to %s at %s. The mean for all %s respondents is %s.",
        tolower(meta$title),
        sprintf("%.2f", s$mean_affairs[i_lo]),
        as.character(s$label[i_lo]),
        sprintf("%.2f", s$mean_affairs[i_hi]),
        as.character(s$label[i_hi]),
        fmt_n(n_all),
        sprintf("%.2f", overall_mean)
      )
    } else {
      i_lo <- which.min(s$affair_rate)
      i_hi <- which.max(s$affair_rate)
      sprintf(
        "Affair rate by %s runs from %s at %s to %s at %s. The rate for all %s respondents is %s.",
        tolower(meta$title),
        fmt_pct1(s$affair_rate[i_lo]),
        as.character(s$label[i_lo]),
        fmt_pct1(s$affair_rate[i_hi]),
        as.character(s$label[i_hi]),
        fmt_n(n_all),
        fmt_pct1(overall_rate)
      )
    }
  })

  output$dist_plot <- renderPlot({
    s <- var_summary()
    meta <- codebook[[input$var]]
    px <- output_px(session, "dist_plot")
    compact <- !is.na(px) && px < 480
    ggplot(s, aes(x = n, y = label)) +
      geom_col(fill = "#0f6e68", width = 0.72) +
      geom_text(
        aes(label = scales::comma(n)),
        hjust = -0.15,
        size = 3.4,
        colour = "#1d2a27"
      ) +
      scale_x_continuous(expand = expansion(mult = c(0, 0)), labels = scales::comma) +
      coord_cartesian(xlim = c(0, max(s$n) * 1.22), clip = "off") +
      labs(
        title = wrap_plot_text(
          paste("People at each level of", tolower(meta$title)),
          px
        ),
        subtitle = wrap_plot_text(
          sprintf("%s recorded levels in the full sample", nrow(s)),
          px
        ),
        x = "People"
      ) +
      theme_affairs(compact)
  }, res = 120, alt = "Bar chart of how many people are at each level of the selected variable.")

  output$level_plot <- renderPlot({
    s <- var_summary()
    meta <- codebook[[input$var]]
    px <- output_px(session, "level_plot")
    compact <- !is.na(px) && px < 480
    if (identical(input$metric, "mean")) {
      ggplot(s, aes(x = mean_affairs, y = label)) +
        geom_vline(
          xintercept = overall_mean,
          linetype = "22",
          linewidth = 0.45,
          colour = "#a34b32"
        ) +
        geom_col(fill = "#315c7a", width = 0.72) +
        geom_text(
          aes(label = sprintf("%.2f", mean_affairs)),
          hjust = -0.15,
          size = 3.4,
          colour = "#1d2a27"
        ) +
        scale_x_continuous(expand = expansion(mult = c(0, 0))) +
        coord_cartesian(
          xlim = c(0, max(s$mean_affairs, overall_mean) * 1.28),
          clip = "off"
        ) +
        labs(
          title = wrap_plot_text(
            paste("Mean affairs measure by", tolower(meta$title)),
            px
          ),
          subtitle = wrap_plot_text(
            sprintf(
              "Dashed line: mean for all respondents (%.2f), zeros included",
              overall_mean
            ),
            px
          ),
          x = "Mean of the affairs measure"
        ) +
        theme_affairs(compact)
    } else {
      ggplot(s, aes(x = affair_rate, y = label)) +
        geom_vline(
          xintercept = overall_rate,
          linetype = "22",
          linewidth = 0.45,
          colour = "#1d2a27"
        ) +
        geom_col(fill = "#a34b32", width = 0.72) +
        geom_text(
          aes(label = sprintf("%.1f%%", 100 * affair_rate)),
          hjust = -0.15,
          size = 3.4,
          colour = "#1d2a27"
        ) +
        scale_x_continuous(
          expand = expansion(mult = c(0, 0)),
          labels = scales::label_percent(accuracy = 1)
        ) +
        coord_cartesian(
          xlim = c(0, max(s$affair_rate, overall_rate) * 1.28),
          clip = "off"
        ) +
        labs(
          title = wrap_plot_text(
            paste("Affair rate by", tolower(meta$title)),
            px
          ),
          subtitle = wrap_plot_text(
            sprintf(
              "Share with affairs > 0. Dashed line: all respondents (%s)",
              fmt_pct1(overall_rate)
            ),
            px
          ),
          x = "Affair rate"
        ) +
        theme_affairs(compact)
    }
  }, res = 120, alt = "Bar chart of the affair rate or the mean affairs measure at each level of the selected variable.")

  output$level_table <- renderTable({
    s <- var_summary()
    data.frame(
      Level = as.character(s$label),
      Code = fmt_num(s$level),
      People = fmt_n(s$n),
      `Share of sample` = fmt_pct1(s$n / n_all),
      `Affair rate` = fmt_pct1(s$affair_rate),
      `Mean affairs measure` = sprintf("%.3f", s$mean_affairs),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  }, striped = TRUE, hover = TRUE, spacing = "s", width = "100%", align = "l")

  output$balance_plot <- renderPlot({
    px <- output_px(session, "balance_plot")
    compact <- !is.na(px) && px < 480
    bal <- data.frame(
      label = factor(
        c("No affair  ·  affairs = 0", "At least one  ·  affairs > 0"),
        levels = c("At least one  ·  affairs > 0", "No affair  ·  affairs = 0")
      ),
      n = c(n_none, n_affair)
    )
    ggplot(bal, aes(x = n, y = label)) +
      geom_col(aes(fill = label), width = 0.62, show.legend = FALSE) +
      geom_text(
        aes(label = sprintf("%s  (%s)", scales::comma(n), fmt_pct1(n / n_all))),
        hjust = -0.08,
        size = 3.6,
        colour = "#1d2a27"
      ) +
      scale_fill_manual(values = c(
        "No affair  ·  affairs = 0" = "#0f6e68",
        "At least one  ·  affairs > 0" = "#a34b32"
      )) +
      scale_x_continuous(expand = expansion(mult = c(0, 0)), labels = scales::comma) +
      coord_cartesian(xlim = c(0, max(bal$n) * 1.38), clip = "off") +
      labs(
        title = wrap_plot_text("Class balance of had_affair", px),
        subtitle = wrap_plot_text(
          "The modeling target. Built from the affairs column; a positive value of any size is class 1.",
          px
        ),
        x = "People"
      ) +
      theme_affairs(compact)
  }, res = 120, alt = "Two bars comparing people with no affair and people with at least one.")

  output$measure_plot <- renderPlot({
    px <- output_px(session, "measure_plot")
    compact <- !is.na(px) && px < 480
    ggplot(pos, aes(x = affairs)) +
      geom_histogram(bins = 28, fill = "#315c7a", colour = "white", linewidth = 0.2) +
      labs(
        title = wrap_plot_text("Original affairs measure, where it is above zero", px),
        subtitle = wrap_plot_text(
          sprintf(
            "%s people at exactly 0 are omitted so the positive measure is visible. Mean above zero: %.2f. Maximum: %.2f.",
            fmt_n(n_none),
            mean(pos$affairs),
            max(pos$affairs)
          ),
          px
        ),
        x = "Affairs measure",
        y = "People"
      ) +
      theme_affairs(compact) +
      theme(
        panel.grid.major.y = element_line(colour = "#ece4d8", linewidth = 0.4),
        panel.grid.major.x = element_blank(),
        axis.title.y = element_text(colour = "#3d4a46")
      )
  }, res = 120, alt = "Histogram of the original affairs measure among people with a value above zero.")

  output$tree_plot <- renderPlot({
    par(bg = "white", mar = c(1, 1, 1, 1))
    rpart.plot(
      fitted_models$tree_fit,
      type = 2,
      extra = 106,
      under = TRUE,
      fallen.leaves = TRUE,
      roundint = FALSE,
      digits = 3,
      clip.right.labs = FALSE,
      box.palette = c("#d7ebe7", "#f3d4c8"),
      branch.col = "#5c6b66",
      split.col = "#1d2a27",
      shadow.col = NA,
      tweak = 1.05
    )
  }, res = 120, alt = "Depth-3 classification tree. Nodes show the fitted class, the training probability of an affair, and the share of training rows.")
}

shinyApp(ui, server)
