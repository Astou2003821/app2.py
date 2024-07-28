import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import streamlit as st
import numpy as np

df = pd.read_csv('Financial_inclusion_dataset.csv')

st.title("Prédiction de l'utilisation d'un compte bancaire")

st.sidebar.title('Menu')
page = ['Description de l-application', 'Nettoyage des données', 'Entrainement du model et visualisation', 'prédiction']
Menu = st.sidebar.radio('Naviguez', page)

if Menu == page[0]:
    st.write("cette application  travaille sur l’ensemble de données « Inclusion financière en Afrique » qui a été fourni dans le cadre de l’inclusion financière en Afrique L’ensemble de données contient des informations démographiques et les services financiers utilisés par environ 33 600 personnes en Afrique de l’Est. Le rôle du modèle ML est de prédire quelles personnes sont les plus susceptibles d’avoir ou d’utiliser un compte bancaire Le terme inclusion financière signifie que les particuliers et les entreprises ont accès à des produits et services financiers utiles et abordables qui répondent à leurs besoins – transactions, paiements, épargne, crédit et assurance – dispensés de manière responsable et durable. Malgré une augmentation de l’inclusion financière en Afrique, le continent enregistre toujours des indicateurs inférieurs au niveau global. L’exclusion financière de la population africaine est surtout liée à des facteurs « involontaires ». Les populations subissent le coût trop élevé des services financiers, l’éloignement géographique trop important des banques et le manque de documents officiels.En ce qui concerne les déterminants microéconomiques, être un homme riche, éduqué et plus âgé augmente la probabilité d’être inclus financièrement, éducation et revenu étant les facteurs les plus influents. L’accès au financement formel est un enjeu majeur pour le développement du continent mais il doit correspondre à une certaine réalité de l’économie. En effet, l’inclusion financière pose la question des dangers de l’endettement excessif, phénomène observé notamment dans le cadre de la microfinance.Pour amorcer une inclusion financière vertueuse, les économies africaines doivent se doter de systèmes financiers solides. Les gouvernements peuvent contribuer à ce phénomène grâce à des politiques publiques visant par exemple à améliorer l’accès aux documents officiels, à développer les voies de communication avec les zones géographiques éloignées, et en régulant le système bancaire émergent.")
    st.write('## Exemple d_inclusion financière en Afrique')
    st.image("astou.png")

elif Menu == page[1]:

    st.write('## Dataset avant nettoyage')
    st.dataframe(df.head())
    

    st.write('## Nettoyage du dataframe')
    
    st.write('## Gerer les valeurs manquantes')
    st.dataframe(df.isnull().sum())

    st.write('## Gerer les valeurs abberantes avec le boxplot')
    plt.figure(figsize=(20, 15))
    sns.boxplot(data=df)
    plt.title('Valeurs aberrantes')
    plt.ylabel('Valeurs')
    st.pyplot(plt)

    st.write('## Retablissement des valeurs abberantes')
    data_winsorized = pd.DataFrame()
    for col in df.select_dtypes(include=['number']):
        data_winsorized[col] = winsorize(df[col], (0.05, 0.05))
    data_winsorized[col]

    st.write('## Encodage des valeurs catégorielles')
    st.dataframe(df.head())
    
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    

    st.write('## Dataframe après nettoyage')
    st.dataframe(df.head())

elif Menu == page[2]:
    st.write('## Encodage de la variable cible et de ses caractéristiques')
    
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    st.dataframe(df.head())

    x = df[["age_of_respondent", "marital_status", "education_level", "job_type"]]
    y = df["bank_account"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

    rf = RandomForestClassifier(n_estimators=15, random_state=12)
    rf.fit(x_train, y_train)

    fig, ax = plt.subplots(figsize=(20, 10))
    tree.plot_tree(rf.estimators_[0], feature_names=df[col].columns.tolist(), filled=True, ax=ax)

    st.pyplot(fig)

    y_pred = rf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy Score: {accuracy}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    

elif Menu == page[3]:
    st.write('## Encodage de la variable cible et de ses caractéristiques')
    
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    st.dataframe(df.head())
    st.write('## Titre de l-application')

    st.title('Application Streamlit de l-inclusion financière en Afrique')

    x = df[["age_of_respondent", "marital_status", "education_level", "job_type"]]
    y = df["bank_account"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

    rf = RandomForestClassifier(n_estimators=15, random_state=12)
    rf.fit(x_train, y_train)

    joblib.dump(rf, 'RandomForestClassifier.pkl')
    joblib.dump(x.columns.tolist(), 'model_columns.pkl')

    def load_model() :
        model = joblib.load('RandomForestClassifier.pkl')
        return model

    model = load_model()

    st.write('## Entrée utilisateur')
    st.header('Entrer les caractéristiques du modèle')
    

    input_1 = st.number_input('Age of Respondent', min_value=0, max_value=100, value=0)
    input_2 = st.selectbox('Marital Status', ['Single/Never Married', 'Married/Living together', 'Divorced/Separated', 'Widowed', 'Don\'t know', 'Other'])
    input_3 = st.selectbox('Education Level', ['No formal education', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training', 'Other/Dont know/RTA', '6'])
    input_4 = st.selectbox('Job Type', ['Farming and Fishing', 'Self employed', 'Formally employed Private', 'Formally employed Government', 'Remittance Dependent', 'Government Dependent', 'Student', 'Informally employed', 'Other Income', 'No Income'])

    st.write('## Convertir les entrées en numpy array pour la prédiction')
    input_data = np.array([[input_1, input_2, input_3, input_4]])
    st.dataframe(input_data)


    st.write('## Encodage one-hot des données d-entrée')
    input_data_encoded = pd.get_dummies(x)
    st.dataframe(input_data_encoded)


    st.write('## Convertir les entrées en numpy array pour la prédiction après les avoir encodées')
    input_data = np.array(input_data_encoded)
    st.dataframe(input_data)

    st.write('## Bouton de prédiction')
    if st.button('Prédire'):
        prediction = model.predict(input_data)
        st.write(f'La prédiction est: {prediction[0]}')
        predictions = model.predict(input_data)
        st.write("Prédictions : ", predictions)
