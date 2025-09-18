# jupyternotebook_ml
# 1. Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Generate Synthetic Student Data
np.random.seed(42)
n = 200
data = {
    'student_id': np.arange(1, n+1),
    'name': [f'Student_{i}' for i in range(1, n+1)],
    'class': np.random.choice(['A', 'B', 'C', 'D'], size=n),
    'comprehension': np.random.randint(40, 100, size=n),
    'attention': np.random.randint(30, 100, size=n),
    'focus': np.random.randint(30, 100, size=n),
    'retention': np.random.randint(25, 100, size=n),
    'engagement_time': np.random.randint(10, 90, size=n)
}
df = pd.DataFrame(data)
# Create assessment score from weighted cognitive skills
df['assessment_score'] = (
    0.3 * df['comprehension'] +
    0.2 * df['attention'] +
    0.2 * df['focus'] +
    0.2 * df['retention'] +
    0.1 * df['engagement_time'] +
    np.random.normal(0, 5, n)
).astype(int)

# 3. Correlation Analysis
corr = df[['comprehension', 'attention', 'focus', 'retention', 'engagement_time', 'assessment_score']].corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

# 4. ML Model: Predict Assessment Score
X = df[['comprehension', 'attention', 'focus', 'retention', 'engagement_time']]
y = df['assessment_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print("Score on test set:", model.score(X_test, y_test))

# 5. Clustering: Learning Personas
kmeans = KMeans(n_clusters=3, random_state=42)
df['persona'] = kmeans.fit_predict(X)
persona_names = {0: 'The Striver', 1: 'The Consistent', 2: 'The Explorer'}
df['persona_name'] = df['persona'].map(persona_names)

# 6. Charts
# Bar: Average skill vs score
avg_skills = df.groupby('class')[['assessment_score', 'comprehension', 'attention', 'focus', 'retention', 'engagement_time']].mean()
avg_skills.plot(kind='bar')
plt.title("Average Skills & Scores by Class")
plt.show()

# Scatter: Attention vs Performance
plt.scatter(df['attention'], df['assessment_score'])
plt.xlabel("Attention")
plt.ylabel("Assessment Score")
plt.title("Attention vs Assessment Score")
plt.show()

# Radar: Individual Student Profile
import plotly.graph_objects as go
student_index = 0
skills = ['comprehension', 'attention', 'focus', 'retention', 'engagement_time']
values = [df.loc[student_index, skill] for skill in skills]
fig = go.Figure(data=go.Scatterpolar(r=values + [values[0]],
    theta=skills + [skills[0]], fill='toself'))
fig.update_layout(title=f"{df.loc[student_index, 'name']} Profile")
fig.show()

# 7. Save Data
df.to_csv("synthetic_student_data.csv", index=False)<img width="478" height="705" alt="otp2" src="https://github.com/user-attachments/assets/9862157f-4dd8-427d-82c0-8172aa6c8e82" />
<img width="907" height="717" alt="otp1" src="https://github.com/user-attachments/assets/b2ead7b5-94ac-46b5-8751-77f19c8b8772" />
