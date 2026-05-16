import pandas
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

y_train = ['恶性','恶性','恶性','恶性','恶性','恶性','良性','良性','良性','良性']

y_pred_A = ['恶性','恶性','恶性','良性','良性','良性','良性','良性','良性','良性']
y_pred_B = ['恶性','恶性','恶性','恶性','恶性','恶性','恶性','恶性','良性','恶性']

label = ['恶性','良性']
df_label = ['恶性(正例)','良性(负例)']

cm_A = confusion_matrix(y_train,y_pred_A,labels=label)
cm_B = confusion_matrix(y_train,y_pred_B,labels=label)

df_A = pandas.DataFrame(cm_A,index=df_label,columns=df_label)
df_B = pandas.DataFrame(cm_B,index=df_label,columns=df_label)
print(f'混淆矩阵A :\n{df_A}')
print(f'混淆矩阵B :\n{df_B}')

print(f'模型A的精确率：{precision_score(y_train,y_pred_A,pos_label="恶性")}')
print(f'模型A的召回率：{recall_score(y_train,y_pred_A,pos_label="恶性")}')
print(f'模型A的F1值：{f1_score(y_train,y_pred_A,pos_label="恶性")}')
print(f'模型B的精确率：{precision_score(y_train,y_pred_B,pos_label="恶性")}')
print(f'模型B的召回率：{recall_score(y_train,y_pred_B,pos_label="恶性")}')
print(f'模型B的F1值：{f1_score(y_train,y_pred_B,pos_label="恶性")}')
