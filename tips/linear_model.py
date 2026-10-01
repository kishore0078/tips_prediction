from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('tips')
df_check = df.copy()

number_col = df_check.select_dtypes(include=['number'])
text_col = df_check.select_dtypes(include=['object','category'])

number_col = number_col.drop(columns = ['tip'])
text_col = pd.get_dummies(text_col,drop_first=True)

scaler = StandardScaler()

scaled_no = scaler.fit_transform(number_col)
scaled_no = pd.DataFrame(scaled_no,columns=number_col.columns,index= number_col.index)

final_df = pd.concat([text_col,scaled_no],axis=1)

X = final_df
y = df_check['tip']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
result = model.predict(X_test)


score = r2_score(y_test,result)
print(f"R2 Score of this model is {score*100:.2f}%")

plt.figure(figsize=(6,4))
plt.scatter(y_test,result,alpha=0.7,color='red')
plt.title("Real data vs Predicted data")
plt.xlabel('Real data')
plt.ylabel('predicted data')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', linestyle='--')
plt.show()