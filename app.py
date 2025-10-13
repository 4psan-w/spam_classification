import streamlit as st
import joblib
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from regex import sub
from mysql.connector import connect


ps_stemmer = PorterStemmer()

class SpamClassifier:
   
    #Initialisation of the Model 

    # PreTrained Model and The TFIDF vectoriser

    def __init__(self):
        self.model = joblib.load("spam_model.pkl")
        self.vectorizer = joblib.load("vectoriser.pkl")
        
        # also initialised the Stopwords cause we need it
        self.stop_words = set(stopwords.words('english'))

        
    # Function Used to build the connection with the Database

        #Use case : 
            #-> Feedbacks and corrections 
        

    def sql_connection(self):
        usr_passwd = 'FUCKOFF69'
        connection = connect(
            host = 'localhost' ,
            user = 'root',
            password = usr_passwd,
            database ='Project',
            port = 3306 
        )

        # Returns the Connection info and the Cursor to execute the Queries
        return connection , connection.cursor()


    #Controlling the Actual SQL query
    def Correction(self,text,spam,corrected):
        conn , cursor = self.sql_connection()
        # corrected = st.radio("Are you Satisfied With the Output ?\if Not What should it be? " ,['Spam' , "Not Spam"])
        # QUERY
        corrected = 1 if corrected == 'Spam' else 0
        cursor.execute(f'insert into Correction(text , Model_Predicted , User_correction) values(%s,%s,%s)'
                       ,(text ,spam ,corrected))
        # Commiting the Info.
        conn.commit()
        # conn.close()

    
    #method for a mild introduction
    def introduction(self):
        st.title("Spam Classifier")
        st.write("This application classifies messages as 'Spam' or 'Ham' (not spam) using a pre-trained Naive Bayes model.")
    
    # Text Cleaning
    def Clean_texts(self, text):
        text = sub(r'[^a-zA-Z]' , " " , text)
        text = word_tokenize(text)
        text_corpus = [ps_stemmer.stem(words.lower()) for words in text if words not in self.stop_words]
        return " ".join(text_corpus)
    
    # Vectorisation
    def preprocess_texts(self , text):
        cleaned_text = self.Clean_texts(text)

        vectorised_text = self.vectorizer.transform([cleaned_text])
        return vectorised_text

    
    # Model Prediction
    def isspam(self, text):
        processed_text = self.preprocess_texts(text)
        spam = self.model.predict(processed_text)
        return spam


    # Main function for the overall control
    def main(self):
        self.introduction()
        user_input = st.text_area("Enter your message here:")
        if st.button("Submit"):
            if(user_input):
                spam=self.isspam(user_input)
                spam=spam[0]    
                if spam:
                    st.error("Its Concluded as a Spam Mesage")
                else:
                    st.success("Its Concluded as a Not a Spam Message")
            else:
                st.error("Message Cannot Be empty") 
            # st.write(self.Clean_texts(user_input))
        
        self.Correction(user_input,spam)


if __name__ == "__main__":
    app = SpamClassifier()
    app.main()
    # app.predict(["Congratulations! You've won a free ticket to Bahamas. Click here to claim."])