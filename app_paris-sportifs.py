#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date:  16/03/2024

@author: Lionel, Nicolas, Sebastione et François
"""

# Core Pkg
import streamlit as st

# Custom modules

def bases_streamlit():
    # streamlit features
    # TEXT
    # titre
    st.title("Projet Datasicentest")

    # texte
    st.text("Les paris sportifs, prédiction des résultats des matchs de football")

    # header \ subheader
    st.header("This is a Header")
    st.subheader('This a Subheader')

    # MARKDOWN
    st.markdown("#### This is a markdown")

    # Link
    st.markdown('[Click here to access Google](https://google.com)')
    
    
    # Alert text
    st.write("Alert text")

    st.success("Ceci est un succès")
    st.info("Information")
    st.warning("Be careful, these data are only for 2021")
    st.error("The code contains an error")
    
    ## MEDIA
    # Image
    # import Image function
    from PIL import Image
    st.write("Opening an image:")

    # open an image
    img = Image.open("OIP (2).jpeg")

    # Plot the image
    st.image(img, caption="DataScientest Logo")
    
    # Audio
    #audio_file = open('name_of_file.ext', "rb")
    #audio_bytes = audio_file.read()
    #st.audio(audio_bytes, format="audio/mp3")
    
    # Video with URL
    st.subheader("A Youtube video:")
    st.video(data="https://www.youtube.com/watch?v=SNNK6z03TaA")
    
    ### WIDGET
    st.subheader("Let's talk about widgets")

    # Bouton
    st.button("Press")
    
    # Other button
    
    #result = st.button("press me please : bouton")
    #if result : 
        #st.text('you have succeeded! ')
    
    # getting interaction button
    if st.button("Press Me"):
        st.success("this is a success!")
        st.error("Error")
    
    # Checkbox
    if st.checkbox("Hide & seek"):
        st.success("showing")
    
    # Radio
    
    gender = st.radio("Select a gender", ["Man", "Woman"])
    
    st.text(gender)
    
    
    if gender == 'Man':
        st.text("This is a Man")
        
    if gender == 'Woman':
        st.text("This is a woman")
    else: 
        st.info("No gender selected yet")
    
    # Select
    job = st.selectbox("Your Job", ["Data Scientist", "Dentist", "Doctor"])
    
    
    if job == "Data Scientist" : 
       
        st.info("You are a Data Scientist")
        
    else : 
        
        st.warning("Pas de metier sélectionné")
        

    
    # Multiselect
    variables = st.multiselect("list de variables",
                                    ["Iphone", "Lemon", "Orange"])
    
    st.text(variables)
    
    # Text imput
    name = st.text_input("your name", "your name here")
    st.text(name)
    st.text(name[:2])
    
    # Number input
    age = st.number_input("Age", 5, 100)
    
    st.text(age*2)
    
    # text area
    message = st.text_area("Enter your message")
    
    # Slider
    level = st.slider("select the level", 0, 100)
    
    
    st.write(level*2)
    
    # Ballons
    if st.button("Press me again"):
        st.write("Yesss, you'r ready!")
        st.balloons()
        
    import pandas as pd
        
    df = pd.read_csv('titanic.csv')
    
    st.dataframe(df.head())
    
    variables = st.multiselect("list de variables",
                                    ["pclass", "survived", "age"])
    
    st.dataframe(df[variables])
    
    df.dropna()
    
    import seaborn as sns 
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.countplot(x = 'pclass' , data = df)
    st.pyplot()
    
    sns.heatmap(df.corr())
    st.pyplot()
    
    from sklearn.model_selection import train_test_split 
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestRegressor
    
    df = df.dropna()

    # Drop some columns
    df = df.drop(['sex', 'title', 'cabin', 'embarked'], axis = 1)

    # Select the target
    y = df['survived']

    

    
    

bases_streamlit()