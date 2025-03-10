from numpy import float64, int64
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

def toCSV(source="./data/data.xlsx",dest="./data/data.csv"):
    df = pd.read_excel(source)
    df.to_csv(dest, index=False)

def printInfo(df):
    print(df.shape)
    print(df.iloc[0,:])
    print(df.dtypes)
    print("="*10)

def str2intList(strList):
    pattern = r"^\s*(\d+\.?\d*)\s*[Xx×]\s*(\d+\.?\d*)\s*$"
    n1List=[]
    n2List=[]
    for string in strList:
        match = re.match(pattern, string)
        num1, num2 = match.groups()
        n1List.append(num1)
        n2List.append(num2)
    return n1List,n2List

def stripStr(string):
    res = re.sub(r'(囊胚|囊)*(\d+)(囊胚|囊)*', r'\2', string)
    return res


def convertType(df):
    n1List,n2List=str2intList(df['孕囊'])
    df=df.drop(columns=["孕囊"])
    df["n1"]=n1List
    df["n2"]=n2List
    df['n1'] = df['n1'].astype(int64)
    df['n2'] = df['n2'].astype(int64)
    df["ET天"]=df["ET天"].map(stripStr).astype(int64)
    df['胚芽'] = pd.to_numeric(df['胚芽'], errors='coerce')
    df = df.dropna(subset=['胚芽'])
    df['卵黄囊'] = pd.to_numeric(df['卵黄囊'], errors='coerce')
    df = df.dropna(subset=['卵黄囊'])
    return df

def removeOutliersIQR(df,delta=1.5):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - delta * IQR
    upper_bound = Q3 + delta * IQR
    return df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

def process(df):
    df=convertType(df)
    df=removeOutliersIQR(df)
    X=df.drop(columns=["标签","序号"])
    y=df["标签"]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return df,X,y

def train_report(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = SVC(kernel='rbf', probability=True, random_state=42)  
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
    print("Classification Report:\n", report)

    auc = roc_auc_score(y_test, y_prob)
    print("AUC Score:", auc)


def main():
    df=pd.read_csv("./data/data.csv")
    printInfo(df)
    df,X,y=process(df)
    printInfo(df)
    print(X,y)
    train_report(X,y)

    
if __name__=="__main__":
    main()

