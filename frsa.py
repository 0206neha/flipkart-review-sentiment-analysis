# for sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# for word cloud and removing stopwords form reviews
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

# For stemming of data
from nltk.stem.porter import PorterStemmer

# Get the list of stopwords
from nltk.corpus import stopwords

# For uding dash components- web based dashboard
from dash import Dash, html, dcc,dash_table

# For plotting graphs
import matplotlib.pyplot as plt
import plotly.express as px

# for data preprocessing
import pandas as pd
import numpy as np
import re
# nltk.download('vader_lexicon')

sentiments = SentimentIntensityAnalyzer()
df = pd.read_csv("flipkart.csv")

# remove missing values
df=df.dropna()

df['clean_text'] = df['reviewText'].str.replace("[^a-zA-Z]"," ")
print(df['clean_text'])

# remove short words with length lesss than 3
df['clean_text'] = df['clean_text'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
df.head()

tokenized_text = df['clean_text'].apply(lambda x: x.split())
tokenized_text = tokenized_text.apply(lambda sentence: [word for word in sentence if word not in stopwords.words('english')])
print(tokenized_text.head())

# nltk.download('wordnet')
stemmer = PorterStemmer()

# input word to be stemmed
word = "problems"

# apply stemming
stemmed_word = stemmer.stem(word)

# print the stemmed word
print(stemmed_word)

stemmer = PorterStemmer()
tokenized_text = tokenized_text.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
tokenized_text.head(50)

def get_sentiment(text):
    score = sentiments.polarity_scores(text)
    if score['compound'] > 0:
        return 'positive'
    elif score['compound'] < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to each review in the dataset
df['sentiments'] = df['reviewText'].apply(get_sentiment)

# Print the count of positive, negative, and neutral reviews
print(df['sentiments'].value_counts())

sentiment_count=df['sentiments'].value_counts()
gr=px.bar(sentiment_count,x=sentiment_count.index,y=sentiment_count.values, labels={'x': 'Sentiment', 'y': 'Number of Reviews'})
gr.show()

reviews=" ".join([sentence for sentence in df['clean_text']])
positive_words=[]
negative_words=[]
neutral_words=[]
splitted=reviews.split()
for word in splitted:
    senti=get_sentiment(word)
    if senti=="positive":
        if word not in positive_words:
            positive_words.append(word)
        else:
            continue
    elif senti=="negative":
        if word not in negative_words:
            negative_words.append(word)
        else:
            continue
    else:
        if word not in neutral_words:
            neutral_words.append(word)
        else:
            continue

positive_words_string=" ".join(positive_words)
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(positive_words_string)

# plot the graph
plt.figure(figsize=(9,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


negative_words_string=" ".join(negative_words)
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(negative_words_string)

# plot the graph
plt.figure(figsize=(9,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

frequent_words=dict()
frequents=[]

print(len(splitted))
for word in splitted:
    if word in frequent_words:
        continue
    cnt=splitted.count(word)
    
    if cnt<3:
        continue
    frequent_words[word]=cnt
    frequents.append(word)

freq_words_string=" ".join(frequents)
wordcloud_freqs = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(freq_words_string)
print(type(wordcloud_freqs))
# plot the graph
plt.figure(figsize=(9,5))
plt.imshow(wordcloud_freqs, interpolation='bilinear')
plt.axis('off')
plt.show()

pos_freq=dict()
freq_pos=[]
for word in positive_words:
    if word in pos_freq:
        continue
    cnt=splitted.count(word)
    if cnt<2:
        continue
    pos_freq[word]=cnt
    freq_pos.append(word)
#sort dictionary in descending order,x[1]=value of pairs in dict  
pos_freq=dict(sorted(pos_freq.items(), key=lambda x: x[1],reverse=True))

neg_freq=dict()
freq_neg=[]
for word in negative_words:
    if word in neg_freq:
        continue
    cnt=splitted.count(word)
    if cnt<2:
        continue
    neg_freq[word]=cnt
    freq_neg.append(word)
    
# sort dictionary in descending order ,x[1]=value in pair
neg_freq=dict(sorted(neg_freq.items(), key=lambda x: x[1],reverse=True))

# Frequently occuring negative words

freq_negative_words_string=" ".join(freq_neg)
wordcloud_freqneg = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(freq_negative_words_string)

# Frequently occuring positive words

freq_positive_words_string=" ".join(freq_pos)
wordcloud_freqpos = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(freq_positive_words_string)


# print(df['reviewTime'])
df['Year']=[dates.split('-')[2] for dates in df['reviewTime']]

months_list=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


df['Month']=[months_list[int(dates.split('-')[1])-1] for dates in df['reviewTime']]
df['Months']=[dates.split('-')[1] for dates in df['reviewTime']]
df['Month'] = pd.Categorical(df['Month'], categories=months_list, ordered=True)
df = df.sort_values(['Year', 'Month'])

print(df['Year'])
print(df[0:50])

sentiment_counts = df.groupby(['Year', 'Month', 'sentiments']).size().reset_index(name='count')
print(sentiment_counts)
# pivot the dataframe to get counts of each sentiment in each month
sentiment_counts_pivot = pd.pivot_table(sentiment_counts, values='count', index=['Year', 'Month'], columns=['sentiments'], fill_value=0)
print(sentiment_counts_pivot)

total_counts = sentiment_counts.groupby(['Year', 'Month'])['count'].sum().reset_index()

# Merge the total counts back into the original dataframe
df_new = sentiment_counts.merge(total_counts, on=['Year', 'Month'], suffixes=('', '_total'))

# Calculate the percentage of each sentiment for each month
df_new['percentage'] = df_new['count'] / df_new['count_total'] * 100

print(df_new.head(20))


from dash.dependencies import Input, Output
table= pd.DataFrame({'word': list(neg_freq.keys()), 'count': list(neg_freq.values())})


options = [
    {'label': 'Frequently Occurring Positive Words', 'value': 'freq_pos'},
    {'label': 'Frequently Occurring Negative Words', 'value': 'freq_neg'},
    {'label': 'Frequently Occurring Words in Reviews', 'value': 'freq_all'}
]

# Define the default value for the dropdown
default_option = 'freq_pos'

app = Dash(__name__)

# Graph for count of reviews vs sentiment
gr = {
    'data': [
            {'x':sentiment_count.index , 'y':sentiment_count.values , 'type': 'bar', 'name': 'Count'},    ],
    'layout': {
        'title': 'Counts of Reviews vs Sentiment'
    }
}


# Frequently occuring positive words,wordcloud
wordcloud_freq_pos = {
    
    'data': [
        {
            'type': 'image',
            'x': [0, 1],
            'y': [0, 1],
            'sizex': 2,
            'sizey': 2,
            'sizing': 'stretch',
            'opacity': 1,
            'layer': 'below',
            'source':wordcloud_freqpos.to_image()
        }
    ],
    'layout': {
        'title': 'Frequently Occurring Positive Words',
        'xaxis': {'visible': False},
        'yaxis': {'visible': False},
        'sizing': 'stretch',
        'layer': 'below'
    }
}

# wordcloud for frequently occuring words
wordcloud_frequents = {
    'data': [
        {
            'type': 'image',
            'x': [0, 1],
            'y': [0, 1],
            'sizex': 2,
            'sizey': 2,
            'sizing': 'stretch',
            'opacity': 1,
            'layer': 'below',
            'source':wordcloud_freqs.to_image()
        }
    ],
    'layout': {
        
        'title': 'Frequently Occurring Words in Reviews',
        'xaxis': {'visible': False},
        
        
        'yaxis': {'visible': False}
    }
}

#word cloud for frequently occuring negatve words
wordcloud_freq_neg = {
    'data': [
        {
            'type': 'image',
            'x': [0, 1],
            'y': [0, 1],
            'sizex': 2,
            'sizey': 2,
            'sizing': 'stretch',
            'opacity': 1,
            'layer': 'below',
            'source':wordcloud_freqneg.to_image()
        }
    ],
    'layout': {
        'title': 'Frequently Occurring Negative Words',
        'xaxis': {'visible': False},
        'yaxis': {'visible': False}
    }
}
app.layout = html.Div(children=[
    html.H1(children='Amazon Review Sentiment Analysis', style={'text-align': 'center'}),
    
    dcc.Graph(
        id='example-graph',
        figure=gr
    ),

    # Dropdown to select the word cloud to be displaye
    html.Div([
        html.Label('Select a wordcloud:', style={'font-weight': 'bold','font-size': '20px', 'text-align':'center'}),
        html.Br(),
        dcc.Dropdown(
            id='wordcloud-dropdown',
            options=options,
            value=default_option,
            style={
            'width': '300px',
            'height': '40px',
            'margin-top': '20px'
             }
        ),
        
        html.Br(),
        html.Br(),
        
        # section where word cloud will be displayed
        html.Div(
            id='wordcloud-container',
            children=[
                dcc.Graph(
                    id='wordcloud-graph',
                    figure=wordcloud_freq_pos,
                    style={
                         'width': '1500px',
                         'height': '1200px'
                    }
                    
                )
            ]
        )
    ]),
    
    # Graph for month vs count of reviews-Bar chart
    
    
    html.Div([
    html.H2(children='Sentiments count by month', style={'text-align': 'center'}),

     html.Label('Select Year:', style={'font-weight': 'bold','font-size': '20px', 'text-align':'center'}),

    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': year, 'value': year} for year in df['Year'].unique()],
         placeholder="Select a year",value=2012,style={'width': '300px'}
    ),
            dcc.Graph(id='sentiment-counts-graph',figure={})

]),
    # Graph for month vs count of reviews-Line chart

    html.Div([
     html.H2(children='percentage of sentiments by month', style={'text-align': 'center'}),

     html.Label('Select Year:', style={'font-weight': 'bold','font-size': '20px', 'text-align':'center'}),

    dcc.Dropdown(
        id='year-dropdown2',
        options=[{'label': year, 'value': year} for year in df['Year'].unique()],
         placeholder="Select a year",value=2012,style={'width': '300px'}
    ),
            dcc.Graph(id='sentiment-counts-graph-line',figure={})

]),
    
    html.Div([
     html.Label('Select a sentiment:', style={'font-weight': 'bold','font-size': '20px', 'text-align':'center'}),

    dcc.Dropdown(
        id='sentiment-dropdown',
        options=[{'label': sentiment, 'value': sentiment} for sentiment in df_new['sentiments'].unique()],
         placeholder="Select a sentiment",value='Positive',style={'width': '300px'}
    ),
            dcc.Graph(id='sentiment-percent-graph-line',figure={})

]),
    
    
    # Table having word and its frequency
    html.H2(children='Frequently occuring negative words'
            ,style={
                'text-align':'center',
                 'font-size': '20px',
                 'margin-top': '40px'

            }),
    dash_table.DataTable(id='table', columns=[{'name': i, 'id': i} for i in table.columns], data=table.to_dict('records'),  style_cell={'textAlign': 'center', 'fontSize': '18px', 'padding': '10px', 'font-family': 'Verdana'})


])

@app.callback(
    Output('sentiment-counts-graph', 'figure'),
    Input('year-dropdown', 'value')
)
def update_sentiment_counts_graph(year):
    
    # This is for stack graph
    
#     filtered_df = sentiment_counts_pivot[sentiment_counts_pivot.index.get_level_values('Year') == year].reset_index()
#     fig = px.bar(filtered_df, x='Month', y=['positive', 'neutral', 'negative'], color_discrete_sequence=['green', 'gray', 'blue'], barmode='stack', title='Sentiment Distribution',
#              labels={'Month': 'Month', 'value': 'Number of Occurrences', 'variable': 'Sentiment'},
#              category_orders={'Month': ['01', '02', '03', '04', '05', '06', '07', '08', '09','10','11','12']})
#     return fig


    df_filtered = df_new[df_new['Year'] == year]
    color_map = {'negative': 'red', 'neutral': 'gray', 'positive': 'green'}

    fig = px.bar(df_filtered, x='Month', y='percentage', color='sentiments', 
             barmode='group', color_discrete_map=color_map,
             category_orders={'sentiments': ['negative', 'neutral', 'positive'],'Month':['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']}, 
             labels={'sentiments': 'Sentiments', 'percentage': 'sentiment count percentage per month', 'count': 'Count','Month':'Month'})

    return fig


@app.callback(
    Output('sentiment-counts-graph-line', 'figure'),
    Input('year-dropdown2', 'value')
)
def update_sentiment_counts_graph_line(year):
#     df_filtered = sentiment_counts[sentiment_counts['Year'] == year]
    df_filtered = df_new[df_new['Year'] == year]

    color_map = {'negative': 'red', 'neutral': 'blue', 'positive': 'green'}
    fig = px.line(df_filtered, x='Month', y='percentage', color='sentiments', 
                 line_group='sentiments', color_discrete_map=color_map,
                 category_orders={'sentiments': ['negative', 'neutral', 'positive'],'Month':['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']}, 

                 labels={'sentiments': 'Sentiments', 'percentage': 'sentiment count percentage per month', 'Month': 'Month'})

    fig.update_traces(mode='lines+markers')

    return fig

# Graph with sentiments in dropdown,x axis as month,y axis as percentage and lines as year
@app.callback(
    Output('sentiment-percent-graph-line', 'figure'),
    Input('sentiment-dropdown', 'value')
)
def yearly_line_graph(sentiment):
    df_filtered = df_new[df_new['sentiments'] == sentiment]
    years_list=[year for year in df_new['Year'].unique()]
    color_map = {'negative': 'red', 'neutral': 'blue', 'positive': 'green'}
    fig = px.line(df_filtered, x='Month', y='percentage', color='Year', 
                 line_group='Year',
                 category_orders={'sentiments': ['negative', 'neutral', 'positive'],'Month':['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']}, 
                 labels={'sentiments': 'Sentiments', 'percentage': 'sentiment count percentage per month', 'Month': 'Month'})

    fig.update_traces(mode='lines+markers')

    return fig


    
    
@app.callback(
    Output('wordcloud-container', 'children'),
    Input('wordcloud-dropdown', 'value')
)
def update_wordcloud(value):
    if value == 'freq_pos':
        return dcc.Graph(
            id='wordcloud-graph',
            figure=wordcloud_freq_pos
        )
    elif value == 'freq_neg':
        return dcc.Graph(
            id='wordcloud-graph',
            figure=wordcloud_freq_neg
        )
    elif value == 'freq_all':
        return dcc.Graph(
            id='wordcloud-graph',
            figure=wordcloud_frequents
        )
if __name__ == '__main__':
    app.run_server()