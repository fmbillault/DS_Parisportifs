#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date:  16/03/2024

@author: Lionel, Nicolas, Sebastion et François
"""

# Core Pkg
import streamlit as st

# Custom modules

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import json


# Load the original image
original_image = Image.open("logo_datascientest.png")

# Create a white background image with the same size as the original image
background = Image.new("RGBA", original_image.size, (255, 255, 255, 255))

# Composite the original image on top of the white background
composite_image = Image.alpha_composite(background.convert("RGBA"), original_image.convert("RGBA"))

# Save the composite image with a white background
composite_image.save("logo_datascientest_with_white_bg.png")

#Charegement des df
df=pd.read_csv("L1_main_stats_cotes_1.csv")
df_L1=pd.read_csv("Ligue1.csv")
df_rou=pd.read_csv("last_avant_split.csv")

# Function to display home page
def display_home():
    
    st.markdown("### Bienvenue! 🖐🏼 🎲 🔮 ⚽")
    # Add an image to the main page
    st.image("ballon_foot.jpg", width=800)

# Function to display  Les Données page
def display_les_donnees():
    st.markdown("### Les Données")
    st.markdown("### Dataframe sur l'historique des equipes de L1 et leurs ELO")
    st.dataframe(df.head())
    st.write("shape du dataframe : ",df.shape)
    st.dataframe(df.describe())
    st.markdown("### Dataframe sur les moyennes roulantes")
    st.dataframe(df_rou.head())
    st.write("shape du dataframe : ",df_rou.shape)
    st.dataframe(df_rou.describe())

    st.markdown("### Exemple de fichier JSON pour les stats joueurs")

    # JSON fichier joueur
    json_file_name = "joueurs.json"

    try:
        # Read the JSON file with UTF-8 encoding
        with open(json_file_name, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        # Display JSON data using Streamlit's JSON viewer
        st.json(data)

    except FileNotFoundError:
        st.error(f"Error: File '{json_file_name}' not found.")
    
       # Display the separation line with turquoise color
    st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)

    st.markdown("### Gestion des sous blocs du json dans le CSV" )
    st.image("Jsontocsv.png", use_column_width=True)
  


# Function to display the Data Vis page
def display_datavisulalisation():
    st.markdown("### Data visualisation")
    pd.set_option('display.max_columns', 500)

    #Partie Analyse - Répartition des resultats de fin de match
    value_counts = df['FTR'].value_counts()

    
    explode = [0.1] * len(value_counts)
    st.markdown("### Répartition des valeurs de la variable cible FTR dans le dataset")
                      
    fig1 = plt.figure(figsize=(9,6))
    plt.pie(value_counts, labels=value_counts.index, autopct="%.1f%%", explode=explode, shadow=True, startangle=90)
    plt.title('Resultats (H: Home Win, A : Away Win, D: Draw)')
    st.pyplot(fig1)

      # Display the separation line with turquoise color
    st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)

    # Partie Analyse - Taux de bonne prédiction des bookmakers
    st.markdown("### Répartition de prédictions des bookmakers")
    df_bkm_predictions = df[['FTR','bkm_prediction']].copy()
    df_bkm_predictions['bkm_is_right'] = df_bkm_predictions.apply(lambda row: 'Good bkm prediction' if (row['FTR']=='H' and row['bkm_prediction'] == 1)|(row['FTR'] == 'D' and row['bkm_prediction'] == 0)|(row['FTR'] == 'A' and row['bkm_prediction'] == 2) else 'Bad bkm prediction', axis=1)

    counts = df_bkm_predictions['bkm_is_right'].value_counts()
    fig2 = plt.figure(figsize=(4,4))
    plt.pie(counts, labels=counts.index, autopct='%1.2f%%', startangle=140)
    plt.title('Répartition de prédictions des bookmakers')
    plt.axis('equal')                 
    st.pyplot(fig2)

  # Display the separation line with turquoise color
    st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)

    st.markdown("### Analyses des stats d\'équipes")
    st.image("ftg.png", width=800)
    st.image("shots.png", width=800)
    st.image("fouls.png", width=800)

      # Display the separation line with turquoise color
    st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)
    st.markdown("### Analyses des stats des joueurs en fonction de leur positions")
    st.image("stat_joueurs.png", width=800)

# Function to display the Modélisation page
def display_modelisation():
    st.markdown("### Modélisations")
    

    features_matrix = df_rou.drop(columns=['FTR'])
    features_matrix = features_matrix.drop(columns=['B365H','B365A'])
    #features_matrix = features_matrix.drop(columns=['B365H','B365D','B365A','bkm_prediction'])
    target = df_rou['FTR']

    # Load the XGB trained model
    bst_rf = joblib.load('rf_classifier.joblib')


    X_train, X_test, y_train, y_test = train_test_split(features_matrix, target, test_size=0.2, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    X_valid_scaled = pd.DataFrame(scaler.transform(X_valid), columns = X_valid.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

    #
    #bst_rf.fit(X_train_scaled, y_train)
    
    # Perform predictions on the test data
    y_train_pred_rf = bst_rf.predict(X_train_scaled)
    y_valid_pred_rf = bst_rf.predict(X_valid_scaled)
    y_test_pred_rf = bst_rf.predict(X_test_scaled)


    # Calculate accuracy scores
    train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf) * 100
    valid_accuracy_rf = accuracy_score(y_valid, y_valid_pred_rf) * 100
    test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf) * 100

    # Display accuracy scores
    st.markdown("### Classifier RandomForest Classifier")
    st.write("Training Accuracy - RF: {:.2f}%".format(train_accuracy_rf))
    st.write("Validation Accuracy - RF: {:.2f}%".format(valid_accuracy_rf))
    st.write("Test Accuracy - RF: {:.2f}%".format(test_accuracy_rf))

    
    ## Generate the confusion matrix
    confusion_matrix_rf = pd.crosstab(y_test, y_test_pred_rf, rownames=['Classe réelle'], colnames=['Classe prédite par RandomForest'])

    # Display the confusion matrix
    st.write("Confusion Matrix:")
    st.write(confusion_matrix_rf)

    
    # Display classification report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_test_pred_rf))


    # Display the separation line with turquoise color
    st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)

    # Load the XGB trained model
    bst = joblib.load('xgb_classifier.joblib')

    X_train, X_test, y_train, y_test = train_test_split(features_matrix, target, test_size=0.2, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    X_valid_scaled = pd.DataFrame(scaler.transform(X_valid), columns = X_valid.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
    
    # Perform predictions on the test data
    y_train_pred = bst.predict(X_train_scaled)
    y_valid_pred = bst.predict(X_valid_scaled)
    y_test_pred = bst.predict(X_test_scaled)


    # Calculate accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred) * 100
    valid_accuracy = accuracy_score(y_valid, y_valid_pred) * 100
    test_accuracy = accuracy_score(y_test, y_test_pred) * 100

    # Display accuracy scores
    st.markdown("### Classifier XGBoost")
    st.write("Training Accuracy - XGBoost: {:.2f}%".format(train_accuracy))
    st.write("Validation Accuracy - XGBoost: {:.2f}%".format(valid_accuracy))
    st.write("Test Accuracy - XGBoost: {:.2f}%".format(test_accuracy))

    
    ## Generate the confusion matrix
    confusion_matrix = pd.crosstab(y_test, y_test_pred, rownames=['Classe réelle'], colnames=['Classe prédite par XGBoost'])

    # Display the confusion matrix
    st.write("Confusion Matrix:")
    st.write(confusion_matrix)

    # Generate and display the classification report
    # Display classification report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_test_pred))

     # Display the separation line with turquoise color
    st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)

     # Load the RandomForestClassifier model from the joblib file
    model = joblib.load('rf_classifier.joblib')
    features_matrix = df_rou.drop(columns=['FTR'])
    features_matrix = features_matrix.drop(columns=['B365H','B365A'])
   #features_matrix = features_matrix.drop(columns=['B365H','B365D','B365A','bkm_prediction'])
    target = df_rou['FTR']

    X_train, X_test, y_train, y_test = train_test_split(features_matrix, target, test_size=0.2, shuffle=False)

    # Get feature importances from the model
    try:
        feature_importances = model.feature_importances_
    except AttributeError:
        st.error("The loaded model doesn't support feature importances.")
        st.stop()

    # Sort feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Plot the top 20 feature importances
    st.markdown("### Top 20 des variables les plus importantes, Modèle RF classifier")
    top_n = 20
    plt.figure(figsize=(12, 8))
    plt.title("Top {} Feature Importances".format(top_n))
    plt.bar(range(top_n), feature_importances[indices][:top_n], align="center")
    plt.xticks(range(top_n), X_train.columns[indices][:top_n], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Feature Importance")
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot(plt)

     # Display the separation line with turquoise color
    st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)

     # Load the XGBoost model from the joblib file
    model = joblib.load('xgb_classifier.joblib')
    features_matrix = df_rou.drop(columns=['FTR'])
    features_matrix = features_matrix.drop(columns=['B365H','B365A'])
   #features_matrix = features_matrix.drop(columns=['B365H','B365D','B365A','bkm_prediction'])
    target = df_rou['FTR']

    X_train, X_test, y_train, y_test = train_test_split(features_matrix, target, test_size=0.2, shuffle=False)

    # Get feature importances from the model
    try:
        feature_importances = model.feature_importances_
    except AttributeError:
        st.error("The loaded model doesn't support feature importances.")
        st.stop()

    # Sort feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Plot the top 20 feature importances XGBoost
    st.markdown("### Top 20 des variables les plus importantes, Modèle XGBoost classifier")
    top_n = 20
    plt.figure(figsize=(12, 8))
    plt.title("Top {} Feature Importances".format(top_n))
    plt.bar(range(top_n), feature_importances[indices][:top_n], align="center")
    plt.xticks(range(top_n), X_train.columns[indices][:top_n], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Feature Importance")
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot(plt)
# 
def display_preprocessing():
    st.markdown("### Pré-Processing")

     # Display the separation line with turquoise color
    st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)

    st.markdown("### Les étapes du Pré-processing")
    # Open the image

    #st.image("prepro.png", use_column_width=True)
    st.image("PPPart1.png", use_column_width=True)

    st.image("PPPart2.png", use_column_width=True)

   
# Define score_filtered function
def score_filtered(seuil, selected_set_feature, selected_set_target, modele):
    y_pred = modele.predict(selected_set_feature)
    y_proba = modele.predict_proba(selected_set_feature)

    df = pd.DataFrame({'y_pred': y_pred,
                       'y_proba_0': y_proba[:, 0],
                       'y_proba_1': y_proba[:, 1],
                       'y_proba_2': y_proba[:, 2],
                       'y': selected_set_target})

    df['confidence_percentage'] = np.max(y_proba, axis=1)

    df = df[df['confidence_percentage'] > seuil]

    if len(df) == 0:
        return 0
    else:
        return accuracy_score(df['y'], df['y_pred'])

# Function to display Estimation des Gains
def display_estimation():
    st.header("Estimation des Gains")
    selected_model = st.radio("Sélectionner un modèle: ", ("RandomForest Classifier", "XGBoost Classifier"), key="model_selection_v2")


    features_matrix = df_rou.drop(columns=['FTR'])
    features_matrix = features_matrix.drop(columns=['B365H','B365A'])
    target = df_rou['FTR']
    X_train, X_test, y_train, y_test = train_test_split(features_matrix, target, test_size=0.2, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_valid_scaled = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Load selected model
    if selected_model == "RandomForest Classifier":
        model_file = "rf_classifier.joblib"
    elif selected_model == "XGBoost Classifier":
        model_file = "xgb_classifier.joblib"

    modele = joblib.load(model_file)

    # Function to display accuracy graph
    def display_accuracy_graph(model):
        df = pd.DataFrame(columns=['train', 'valid', 'test'])
        for seuil in np.arange(0.28, 0.8, 0.01):
            nouvelle_ligne = {'train': float(score_filtered(seuil, X_train_scaled, y_train, model)),
                              'valid': float(score_filtered(seuil, X_valid_scaled, y_valid, model)),
                              'test': float(score_filtered(seuil, X_test_scaled, y_test, model))}
            nouvelle_ligne = pd.DataFrame([nouvelle_ligne])
            df = pd.concat([df, nouvelle_ligne], ignore_index=True)

        # Plot accuracy graph
        st.write(f"{selected_model} Accuracy Graph")
        fig, ax = plt.subplots()
        for column in df.columns:
            ax.plot(df[column].index + 28, df[column], label=column)
        ax.legend()
        st.pyplot(fig)

    # Display accuracy graph
    st.write(f"Modèle: ")
    display_accuracy_graph(modele)

       # Display the separation line with turquoise color
    st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)

    # choix arbitraire d'un seuil en fonction du visuel du graph au dessus
    #seuil_cut = 0.66
    # Interactive progress bar to select seuil_cut
    seuil_cut = st.slider("Veuillez choisir Seuil :", min_value=0.0, max_value=1.0, value=0.66, step=0.01)
   

    # On affiche les stats obtenus pour le train avec ce nouveau seuil. Juste à titre informatif
    y_train_pred = modele.predict(X_train_scaled)
    y_train_proba = modele.predict_proba(X_train_scaled)
    df_train = pd.DataFrame({'y_train_pred': y_train_pred,
                   'y_train_proba_0': y_train_proba[:, 0],
                   'y_train_proba_1': y_train_proba[:, 1],
                   'y_train_proba_2': y_train_proba[:, 2],
                   'y_train_test': y_train})
    df_train['confidence_percentage'] = np.max(y_train_proba, axis=1)
    df_train = df_train[df_train['confidence_percentage']>seuil_cut]
    st.write(pd.crosstab(df_train['y_train_test'], df_train['y_train_pred'], rownames=['Classe réelle'], colnames=['Classe prédite']))
    score_train_filtered = accuracy_score(df_train['y_train_test'], df_train['y_train_pred'])
    st.write("Le pourcentage de bonne réponse sur les données d'entrainement: ", score_train_filtered)


    # On affiche les stats obtenus pour le valid avec ce nouveau seuil. Juste à titre informatif
    y_valid_pred = modele.predict(X_valid_scaled)
    y_valid_proba = modele.predict_proba(X_valid_scaled)
    df_valid = pd.DataFrame({'y_valid_pred': y_valid_pred,
                   'y_valid_proba_0': y_valid_proba[:, 0],
                   'y_valid_proba_1': y_valid_proba[:, 1],
                   'y_valid_proba_2': y_valid_proba[:, 2],
                   'y_valid_test': y_valid})
    df_valid['confidence_percentage'] = np.max(y_valid_proba, axis=1)
    df_valid = df_valid[df_valid['confidence_percentage']>seuil_cut]
    st.write(pd.crosstab(df_valid['y_valid_test'], df_valid['y_valid_pred'], rownames=['Classe réelle'], colnames=['Classe prédite']))
    score_valid_filtered = accuracy_score(df_valid['y_valid_test'], df_valid['y_valid_pred'])
    st.write("Le pourcentage de bonne réponse sur les données de validation: ", score_valid_filtered)

    # on s'interesse a notre ensemble de test final et sur les resultats obtenus
    y_test_pred = modele.predict(X_test_scaled)
    y_test_proba = modele.predict_proba(X_test_scaled)
    df_test = pd.DataFrame({'y_test_pred': y_test_pred,
                   'y_test_proba_0': y_test_proba[:, 0],
                   'y_test_proba_1': y_test_proba[:, 1],
                   'y_test_proba_2': y_test_proba[:, 2],
                   'y_test_test': y_test,
                   'index_origin' : y_test.index})
    df_test['confidence_percentage'] = np.max(y_test_proba, axis=1)
    max_proba_values = np.max(y_test_proba, axis=1)
    df_test['y_test_proba_max'] = max_proba_values
    df_test = df_test[(df_test['confidence_percentage']>seuil_cut)]
    st.write(pd.crosstab(df_test['y_test_test'], df_test['y_test_pred'], rownames=['Classe réelle'], colnames=['Classe prédite']))
    score_test_filtered = accuracy_score(df_test['y_test_test'], df_test['y_test_pred'])
    st.write("Le pourcentage de bonne réponse sur les données de test: ", score_test_filtered)

    # historique des matchs où l'on aurait misé ainsi que les resultats

    base_cagnotte=0
    mise = 100
    mise_min=100
    mise_max=500
    surete=seuil_cut
    cagnotte = base_cagnotte
    mise_totale = 0
  
    cote = pd.read_csv('cote_for_streamlit.csv', encoding='utf-8', sep=",", parse_dates=['Date'])
    cote = cote[['B365H','B365A','B365D','HomeTeam','AwayTeam','Date','Season']] #.reset_index()
    cote['index_origin'] = cote.index
    df_test = pd.merge(df_test, cote, how="inner", left_on=['index_origin'], right_on=['index_origin'])
    
    df_test["cote_gain"] = np.where(df_test['y_test_test'] == 0, df_test['B365D'],
                   np.where(df_test['y_test_test'] == 1, df_test['B365H'],
                            df_test['B365A']))
    df_test["gain"] = np.where(df_test['y_test_test']==df_test['y_test_pred'], mise*df_test['cote_gain']-mise,(-1)*mise)
    df_test["gain_2_0"] = np.where(df_test['y_test_test']==df_test['y_test_pred'],round((mise_min+(mise_max-mise_min)*(df_test['y_test_proba_max']-surete)/(1-surete))*(df_test['cote_gain']-1)),(-1)*round(mise_min+(mise_max-mise_min)*(df_test['y_test_proba_max']-surete)/(1-surete)))
    df_test["montant_joue"] = round(mise_min+(mise_max-mise_min)*(df_test['y_test_proba_max']-surete)/(1-surete))
    #df_test.sort_values(by=['Date']).tail(11)

       # Display the separation line with turquoise color
    st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)

    # somme des gains si on pari betement 100euros à chaque match

    st.subheader("Estimation des gains, si")
    st.write("Somme des gains si pari à 100 euros sur chaque match, sans stratégie:", df_test['gain'].sum(), "euros")
    st.write("Somme des gains si on pari plus judicieusement le montant à chaque match:", df_test['gain_2_0'].sum(),"euros")
    st.write("Montant total joué pour les gains obtenus:", df_test['montant_joue'].sum(), "euros")

    # Display main variables
    st.subheader("Main Variables")
    st.write("Nombre total de matchs:", len(df_test))

    # Display the DataFrame df_test
    st.subheader("DataFrame df_test")
    new_dftest = df_test.drop(columns=['index_origin', 'confidence_percentage', 'y_test_proba_max' ])
    st.write(new_dftest)

###

# Function to display the Prédiction page
def display_prediction():
    st.header("Les Prédictions")
    # Chargment du 1er modèle Ramodom Forest 
    #with open('randomForest_1.pickle', 'rb') as f:bst_2 = pickle.load(f)
    # Load the saved model
    model = joblib.load('rf_classifier.joblib')
    features_matrix = df_rou.drop(columns=['FTR'])
    features_matrix = features_matrix.drop(columns=['B365H','B365A'])

    target = df_rou['FTR']

    X_train, X_test, y_train, y_test = train_test_split(features_matrix, target, test_size=0.2, shuffle=False)
   
    # Function to predict match outcome
    def predict_match(home_team, away_team, X_train):
        # Select the input data for the home team
        home_data = df.loc[df['HomeTeam'] == home_team].iloc[0]
        
        # Select the input data for the away team
        away_data = df.loc[df['AwayTeam'] == away_team].iloc[0]
        
        # Merge the data for prediction
        input_data = pd.concat([home_data, away_data], axis=1).T.drop(['HomeTeam', 'AwayTeam', 'FTR'], axis=1)  
        
        # Align input data with training data columns
        input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
        
        # Make the prediction
        prediction = model.predict(input_data)[0]
        if prediction == 0:
            return 'Victoire de l\'équipe locale'
        elif prediction == 1:
            return 'Match nul'
        else:
            return 'Victoire de l\'équipe à l\'extérieur'
    
    st.title('Football Match Outcome Predictor - RF Classifier')
    st.write('Sélectionnez l\'équipe qui joue à domicile et l\'équipe qui joue à l\'extérieur pour prédire l\'issue du match')

    # Dropdown menus for home team and away team
    home_team = st.selectbox('Home Team', df['HomeTeam'].unique())
    
    # Ensure away team is not the same as home team
    available_away_teams = [team for team in df['AwayTeam'].unique() if team != home_team]
    away_team = st.selectbox('Away Team', available_away_teams)

    # Predict button
    if st.button('Predict'):
        if home_team.strip() == '' or away_team.strip() == '':
            st.warning('Veuillez sélectionner les deux équipes.')
        else:
            prediction = predict_match(home_team, away_team, X_train)
            st.markdown('<p style="font-size:20px; color: turquoise;">Résultat attendu : {}</p>'.format(prediction), unsafe_allow_html=True)


# Function to display the Equipe Projet page
def display_equipe_projet():
    st.header("Equipe Projet")
    st.write("Lionel, Nicolas, Sebastion et François")
    st.write("Grand Merci Yohan pour son aide 👍")
    # Balloons and progress bar
    st.balloons()

# Set page title
st.title("Les paris sportifs, prédiction des résultats des matchs de football")

# Create a menu in the sidebar

menu = ["Home", "Les Données", "Data visualisation", "Pré-processing", "Modélisation", "Estimation des Gains","Prédiction", "Equipe Projet"]
choice = st.sidebar.selectbox("Menu",menu)

# Display content based on menu choice
if choice == "Home":
    display_home()
elif choice == "Les Données":
    display_les_donnees()
elif choice == "Data visualisation":
    display_datavisulalisation()
elif choice == "Modélisation":
    display_modelisation()
elif choice == "Pré-processing":
    display_preprocessing()
elif choice == "Prédiction":
    display_prediction()
elif choice == "Estimation des Gains":
    display_estimation()
else:
    display_equipe_projet()

# Corps pour toutes les pafes
#st.header("Streamlit Basics")
#st.markdown("### Markdown Example")
#st.text("This is a text widget")


# Add an image to the sidebar
st.sidebar.image("dst-logo.svg", width=50)

# Footer
# Display the separation line with turquoise color
st.markdown("<hr style='border: 2px solid turquoise;'>", unsafe_allow_html=True)
st.markdown("© 2024 DataSciencetest App: Pari Sportif Ligue1")
