#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date:  16/03/2024

@author: Lionel, Nicolas, Sebastione et François
"""

# Core Pkg
import streamlit as st

# Custom modules

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as 

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

df_rou=pd.read_csv("L1_main_stats_cotes_roulante_1.csv")

# Function to display the home page
def display_home():
    st.write("Bienvenue! 🖐🏼")
    # Add an image to the main page
    st.image("ballon.png")

# Function to display the Les Données page
def display_les_donnees():
    st.write("Les Données")
    st.write("Dataframe sur l'historique des equipes de L1 et leurs ELO")
    st.dataframe(df.head())
    st.write(df.shape)
    st.dataframe(df.describe())
    st.write("Dataframe sur les moyennes roulantes")
    st.dataframe(df1.head())
    st.write(df1.shape)
    st.dataframe(df1.describe())
    

# Function to display the Data Vis page
def display_datavisulalisation():
    st.write("Data visualisation page.")
    pd.set_option('display.max_columns', 500)

    # Get value counts of 'FTR' column
    value_counts = df['FTR'].value_counts()

    # Define explode parameter with appropriate length
    explode = [0.1] * len(value_counts)
    st.write("Répartition des valeurs de la variable cible FTR dans le dataset")
                      
    fig1 = plt.figure(figsize=(9,6))
    plt.pie(value_counts, labels=value_counts.index, autopct="%.1f%%", explode=explode, shadow=True, startangle=90)
    plt.title('Resultats (H: Home Win, A : Away Win, D: Draw)')
    st.pyplot(fig1)

    # Partie Analyse - Taux de bonne prédiction des bookmakers
    st.write("Répartition de prédictions des bookmakers")
    df_bkm_predictions = df[['FTR','bkm_prediction']].copy()
    df_bkm_predictions['bkm_is_right'] = df_bkm_predictions.apply(lambda row: 'Good bkm prediction' if (row['FTR']=='H' and row['bkm_prediction'] == 1)|(row['FTR'] == 'D' and row['bkm_prediction'] == 0)|(row['FTR'] == 'A' and row['bkm_prediction'] == 2) else 'Bad bkm prediction', axis=1)

    counts = df_bkm_predictions['bkm_is_right'].value_counts()
    fig2, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.2f%%', startangle=140)
    ax.set_title('Répartition de prédictions des bookmakers')
    ax.axis('equal')                 
    fig2 = plt.figure(figsize=(4,4))
    st.pyplot(fig2)

# Function to display the Modélisation page
def display_modelisation():
    st.write("Modélisation page.")

# Function to display the Prédiction page
def display_prediction():
    st.write("Les Prédictions page.")

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

menu = ["Home", "Les Données", "Data visualisation", "Modélisation", "Prédiction", "Equipe Projet"]
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
elif choice == "Prédiction":
    display_prediction()
else:
    display_equipe_projet()

# Add some widgets to the main page
st.header("Streamlit Basics")
st.markdown("### Markdown Example")
st.text("This is a text widget")


# Add an image to the sidebar
st.sidebar.image("dst-logo.svg", width=50)

# Footer
st.markdown("© 2024 DataSciencetest App: Pari Sportif Ligue1")
