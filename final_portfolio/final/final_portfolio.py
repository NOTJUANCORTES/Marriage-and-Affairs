import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import seaborn as sn
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
# import 
script_dir = Path(__file__).resolve().parent
df = pd.read_csv(script_dir / "_data" / "affairs.csv")

print(df.columns)
print(df.head())

# EDA
#### subplots
ROWS = 2
COLS = 4

fig, ax = plt.subplots(ROWS, COLS, figsize=(12, 8))

ax[0,0].hist(df["rate_marriage"], bins=20)
ax[0,0].set_title("Rate of Marriage")
ax[0,0].set_xlabel("Marriage Rating")
ax[0,0].set_ylabel("Frequency")

ax[0,1].hist(df["age"], bins=20)
ax[0,1].set_title("Age")
ax[0,1].set_xlabel("Age")
ax[0,1].set_ylabel("Frequency")

ax[0,2].hist(df["yrs_married"], bins=20)
ax[0,2].set_title("Years Married")
ax[0,2].set_xlabel("Married Years")
ax[0,2].set_ylabel("Frequency")

ax[0,3].hist(df["children"], bins=20)
ax[0,3].set_title("Number of children")
ax[0,3].set_xlabel("Children")
ax[0,3].set_ylabel("Frequency")


ax[1,0].hist(df["religious"], bins=20)
ax[1,0].set_title("Level of religious commitment (scale: 1-5)")
ax[1,0].set_xlabel("Religious Commitment")
ax[1,0].set_ylabel("Frequency")


ax[1,1].hist(df["educ"], bins=20)
ax[1,1].set_title("Education Level")
ax[1,1].set_xlabel("Education level (years of education)")
ax[1,1].set_ylabel("Frequency")

ax[1,2].hist(df["occupation"], bins=20)
ax[1,2].set_title("Occupation of the Respondent")
ax[1,2].set_xlabel("Occupation")
ax[1,2].set_ylabel("Frequency")

affairs_nonzero = df[df["affairs"] > 0]["affairs"]
ax[1,3].hist(affairs_nonzero, bins=30)
ax[1,3].set_title("Time Spent in Affairs (Months)")
ax[1,3].set_xlabel("Months")
ax[1,3].set_ylabel("Frequency")
ax[1,3].set_xlim(0, 18)
ax[1,3].set_xticks(range(0, 19, 2))

plt.tight_layout()
plt.show()

## Do happier marriages have fewer affairs?
sns.boxplot(x="rate_marriage", y="affairs", data=df)
ma = df.groupby("rate_marriage")["affairs"].mean()
print(ma)
ma.plot(kind="bar", figsize=(10,6), alpha=0.5, color='steelblue', edgecolor='black')
plt.title("Average Time Spent in Affairs by Marriage Satisfaction")
plt.xlabel("Satisfaction with the marriage (scale: 1-5)")
plt.ylabel("Time spent in extramarital affairs (in months)")
plt.show()

## Do more religious people report fewer affairs?
ra = df.groupby("religious")["affairs"].mean()
print(ra)

sns.boxplot(x="religious", y="affairs", data=df)

ra.plot(kind="bar", figsize=(10,6), alpha=0.5, color='steelblue', edgecolor='black')
plt.title("Average Time Spent in Affairs by Level of Religious Commitment")
plt.xlabel("Level of Religious commitment (scale: 1-4)")
plt.ylabel("Time spent in extramarital affairs (in months)")
plt.show()

## Do more people with more children report fewer affairs?
sns.boxplot(x="children", y="affairs", data=df)

ca = df.groupby("children")["affairs"].mean()
print(ca)

ca.plot(kind="bar", figsize=(10,6), alpha=0.5, color='steelblue', edgecolor='black')
plt.title("Average Time Spent in Affairs by Number of children")
plt.xlabel("Number of children")
plt.ylabel("Time spent in extramarital affairs (in months)")
plt.show()

## Do more educated people report fewer affairs?
# educ: Level of education, 9 = grade school, 12 = high school, 
# 14 = some college, 16 = college graduate,
# 17 = some graduate school, 20 = advanced degree

sns.boxplot(x="educ", y="affairs", data=df)
ca = df.groupby("educ")["affairs"].mean()
print(ca)

ca.plot(kind="bar", figsize=(10,6), alpha=0.5, color='steelblue', edgecolor='black')
plt.title("Average Time Spent in Affairs by Level of Education")
plt.xlabel("Education Level ")
plt.ylabel("Time spent in extramarital affairs (in months)")
plt.show()

## Do different occupations report fewer affairs?
occupation_map = {
    1: "Student",
    2: "Agriculture",
    3: "White Collar",
    4: "Teacher/Nurse",
    5: "Managerial",
    6: "Professional"
}

df["occupation_label"] = df["occupation"].map(occupation_map)
#df["occupation_label_husb"] = df["occupation_husb"].map(occupation_map)
ocu_a = df.groupby("occupation_label")["affairs"].mean()
print(ocu_a)

sns.boxplot(x="occupation_label", y="affairs", data=df)
plt.xticks(rotation=45)
plt.show()

## Husband occupational (Occupation of the respondent's husband (for female respondents)

df["occupation_label_husb"] = df["occupation_husb"].map(occupation_map)
ocu_a_husb = df.groupby("occupation_label_husb")["affairs"].mean()
print(ocu_a_husb)

sns.boxplot(x="occupation_label_husb", y="affairs", data=df)
plt.xticks(rotation=45)
plt.title("Average Time Spent in Affairs by Husband Occupation")
plt.xlabel("Husband Occupation")
plt.ylabel("Time spent in extramarital affairs (in months)")
plt.show()

# takinng logistic approach
# How much time was spent in affairs?
# Regression# 0 = No Affair, 1 = Had Affair
df["had_affair"] = (df["affairs"] > 0).astype(int) 
# Binary Classification approach
print(df["had_affair"].value_counts())
df["had_affair"].value_counts(normalize=True)

# using Multiple Linear Regression
X = df[["rate_marriage", "age", "yrs_married", "children", "religious","educ"]]
y = df["had_affair"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

affair_model = LogisticRegression(max_iter=10000)
affair_model.fit(X_train, y_train)
score = affair_model.score(X_test, y_test)

print(f"Score on test data with all features except occupation is: "
      f"{score}")

# confusin matrix: 
y_pred = affair_model.predict(X_test)
print("-----------------------")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Create png of classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print(report_df)
fig, ax = plt.subplots(figsize=(8, 4))

ax.axis('off')

table = ax.table(cellText=report_df.round(3).values, colLabels=report_df.columns,
    rowLabels=report_df.index,
    loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.savefig(script_dir / "classification_report.png", dpi=300, bbox_inches="tight")
plt.show()

# Note:  The logistic regression model achieved 71% accuracy.

# Decison Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print(tree.score(X_test, y_test))

# plot
plt.figure(figsize=(12,8))

plot_tree(tree, feature_names=X.columns, class_names=["No Affair", "Affair"], filled=True)
plt.show()

# Comparison
print("Logistic:", affair_model.score(X_test, y_test))
print("Tree:", tree.score(X_test, y_test))

# barplot
importance = pd.DataFrame({"Feature": X.columns, "Importance": tree.feature_importances_})
print(importance.sort_values("Importance", ascending=False))
