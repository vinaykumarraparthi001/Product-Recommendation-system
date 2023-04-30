from flask import Flask, render_template, request
import pandas as pd

# Import the function that we created earlier for recommending Airbnb listings
from PythonProject_vinay import recommend_listings_text

app = Flask(__name__)

# Define the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the page that displays the recommended Airbnb listings
@app.route('/recommend', methods=['POST'])
def recommend():
    input_text = request.form['text'] # Get the user's input text
    df_recommend = recommend_listings_text(input_text) # Call the recommend function
    
    # Convert the dataframe to a list of dictionaries
    listings = df_recommend.to_dict('records')
    
    # Render the page that displays the recommended listings
    return render_template('recommend.html', listings=listings)

if __name__ == '__main__':
    app.run(debug=True)
