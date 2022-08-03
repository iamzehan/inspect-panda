from ctypes import WinDLL, alignment
from logging import PlaceHolder, raiseExceptions
from optparse import Values
import re
import chardet
import numpy as np # math package
import pandas as pd
from pygments import highlight # dataframe package
import seaborn as sns # plotting package
import streamlit as st # our framework
from scipy import stats # statistical anaylysis package
import matplotlib.pyplot as plt # plotting package
from mlxtend.preprocessing import minmax_scaling #minmax scaling 
import fuzzywuzzy
from fuzzywuzzy import process
from sklearn.cluster import KMeans
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

st.header("Welcome to Inspect Panda 🐼")

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)

    st.pyplot(fig)

choose_source=st.selectbox("Choose source",['Via File Upload','URL'])
if choose_source=='Via File Upload':
    uploaded_file = st.file_uploader("Choose a file",type=['csv','xls'],accept_multiple_files=False,key="fileUploader")
elif choose_source=='URL':
    uploaded_file = st.text_input(label="Enter URL",placeholder='https://wwww.example.com')

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

if uploaded_file!=None and uploaded_file!="":
    try:
        try:
            dataframe = pd.read_csv(uploaded_file)
        except:
            dataframe=pd.read_excel(uploaded_file)
    except:
        with open(uploaded_file,'rb') as rawdata:
            rc=0
            result = chardet.detect(rawdata.read(rc))
            for i in range(rc,100000):
                if result['confidence']<0.5:
                    rc+=10
                elif result['confidence']>0.5: 
                    try:
                        dataframe=pd.read_csv(uploaded_file,encoding=result['encoding'])
                        break
                    except:
                        rc+=10

    select_options=st.sidebar.selectbox("Choose Options",["Select...","Handle Missing Values","Scaling and Normalization","Parsing Dates","Inconsistent Data Entry","Feature Selections"])

#-----------------------------------------HANDLING MISSING VALUES START--------------------------------------------------#
    if select_options=='Handle Missing Values':
        select_missing=st.sidebar.selectbox("Select Options",["Count missing Values","Drop NaNs","Fill Missing Values"])
        if select_missing=="Count missing Values":
            st.markdown("Let's see if we have any missing values 🔎")
            missing_values_count = dataframe.isnull().sum()
            df=missing_values_count.to_frame(name='Missing')
            st.dataframe(df.style.apply(lambda x: ['background-color: #ff2a26' if i>0 else '' for i in x], axis=0))
            total_cells = np.product(dataframe.shape)
            total_missing = missing_values_count.sum()
            # percent of data that is missing
            percent_missing ='{0:.3g}'.format((total_missing/total_cells) * 100)
            st.write(f'About ```{percent_missing}%``` of data is missing...')
        
        elif select_missing=="Drop NaNs":
            st.markdown("Let's see if dropping the Null values do any good!")
            column_or_rows=st.radio("I want to drop",["None","Columns","Rows"],horizontal=True)
            if column_or_rows=="Columns":
                st.write("```Columns Dropped```",color='red')
                columns_with_na_dropped = dataframe.dropna(axis=1)
                st.dataframe(columns_with_na_dropped.head())
                st.write("Columns in original dataset: ```%d``` \n" % dataframe.shape[1])
                st.write("Columns with na's dropped: ```%d```" % columns_with_na_dropped.shape[1])
                columns_lost=abs(dataframe.shape[1] - columns_with_na_dropped.shape[1])
                st.write("Your lose ```%d``` columns" %columns_lost)
            elif column_or_rows=="Rows":
                st.write("```Rows Dropped```")
                rows_with_na_dropped = dataframe.dropna(axis=0)
                st.dataframe(rows_with_na_dropped.head())
                st.write("Rows in original dataset: ```%d``` \n" % dataframe.shape[0])
                st.write("Rows with na's dropped: ```%d```" % rows_with_na_dropped.shape[0])
                rows_lost=abs(dataframe.shape[0] - rows_with_na_dropped.shape[0])
                st.write("Your lose ```%d``` rows" % rows_lost)
        elif select_missing=="Fill Missing Values":
            st.header("Select Columns you want to fill in the data with")
            df = dataframe.copy()
            missing_values_count=pd.DataFrame(dataframe.isnull().sum(), columns=["Null Count"],)
            missing_value_columns=(missing_values_count[missing_values_count["Null Count"]>0]).index.values
            selected_columns=st.multiselect(label="Select Columns",options=missing_value_columns)
            if selected_columns:
                sample=df.loc[:, selected_columns]
                st.dataframe(sample.head())
                st.subheader("Select ```fillna``` Options")
                fill_options=st.selectbox(label="If you want to replace with a specific value choose option 'fillna'",options=[None,"fillna","ffill","bfill"])
                if fill_options==None:
                    st.markdown("For more details on the options above visit this link: [Pandas fillna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)")
                elif fill_options=='fillna':
                    fill_value=st.number_input(label="Fill Value(Don't change this unless you want to:")
                    st.markdown("Applied to:")
                    fill_na_df=dataframe.copy()
                    for sc in selected_columns:
                        fill_na_df[sc]=fill_na_df.loc[:, sc].fillna(value=fill_value)
                        st.markdown(f"```{sc}```")
                    missing_values_count=pd.DataFrame(fill_na_df.isnull().sum(), columns=["Null Count"])
                    preview=missing_values_count.style.apply(lambda x: ['background-color: green' if i in selected_columns else '' for i in x.index.values], axis=0)
                    show_preview=st.button("Preview Missing Values")
                    if show_preview:
                        st.markdown("The following data have been modified and shows there's no missing value in the corresponding columns.")
                        st.dataframe(preview)
                    st.markdown("If you are satisfied with your conversion of the data then hit the download button below to download your newly updated data")
                    st.download_button(label="Download data as CSV",data=convert_df(fill_na_df),file_name=f'zero_filled.csv',mime='csv')
                else:
                    st.markdown("Applied to:")
                    fill_na_df=dataframe.copy()
                    for sc in selected_columns:
                        fill_na_df[sc]=fill_na_df.loc[:, sc].fillna(method=fill_options)
                        st.markdown(f"```{sc}```")
                    missing_values_count=pd.DataFrame(fill_na_df.isnull().sum(), columns=["Null Count"])
                    preview=missing_values_count.style.apply(lambda x: ['background-color: green' if i in selected_columns else '' for i in x.index.values], axis=0)
                    show_preview=st.button("Preview Missing Values")
                    if show_preview:
                        st.markdown("The following data have been modified and shows there's no missing value in the corresponding columns.")
                        st.dataframe(preview)
                    st.markdown("If you are satisfied with your conversion of the data then hit the download button below to download your newly updated data")
                    st.download_button(label="Download data as CSV",data=convert_df(fill_na_df),file_name=f'zero_filled.csv',mime='csv')

            elif selected_columns==[]:
                st.write("```None```")
#-----------------------------------------HANDLING MISSING VALUES END--------------------------------------------------#

#---------------------------------------SCALING AND NORMALIZATION START------------------------------------------------#
    elif select_options=="Scaling and Normalization":
        scaling_normalizing_options=st.sidebar.selectbox(label="Select how you want to scale your data",options=[None,"Scale","Normalize"])
        original_data=dataframe.copy()
        if scaling_normalizing_options==None:
            st.subheader("What is the difference between scaling and normalizing?")
            st.markdown("* in $scaling$, you're changing the range of your data")
            st.markdown("* while in $normalization$, you're changing the shape of the distribution of your data.")
        elif scaling_normalizing_options=="Scale":
            option_columns=[i  for i in original_data.select_dtypes(['float','int'])]
            selected_column=st.selectbox(label="Select Columns",options=[ None,*option_columns])
            st.write(f"Selected: ```{selected_column}```")
            selected_data=pd.DataFrame(columns=[selected_column],data=original_data)
            try:
                if selected_column:
                    scaled_data = minmax_scaling(selected_data, columns=[selected_column])
                    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
                    sns.histplot(selected_data, ax=ax[0], kde=True, legend=False)
                    ax[0].set_title("Original Data")
                    sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
                    ax[1].set_title("Scaled data")
                    st.pyplot(fig)
                else:
                    st.markdown("```None Selected```")
            except:
                st.error(f"Can't Scale `{selected_column}`")
        
        elif scaling_normalizing_options=="Normalize":
            option_columns=[i  for i in original_data if original_data[i].dtype==(float or int)]
            selected_column=st.selectbox(label="Select Columns",options=[ None,*option_columns])
            st.write(f"Selected: ```{selected_column}```")
            if selected_column != None:
                selected_data=pd.DataFrame(columns=[selected_column],data=original_data)
                selected_data=selected_data.dropna()
                selected_data=selected_data.loc[(selected_data!=0).any(axis=1)]
                if st.button("View Data›"):
                    st.dataframe(selected_data)
                try:
                    normalized_data = stats.boxcox(selected_data[selected_column])
                    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
                    sns.histplot(selected_data, ax=ax[0], kde=True, legend=False)
                    ax[0].set_title(f"Original Data: '{selected_column}'")
                    sns.histplot(normalized_data, ax=ax[1], kde=True, legend=False)
                    ax[1].set_title(f"Normalized Data: '{selected_column}'")
                    st.pyplot(fig)  
                except:
                    st.error(f"Can't normalize ```'{selected_column}'```")
                st.markdown("Remember we are not considering values that are less than or equal to zero!")          

#---------------------------------------SCALING AND NORMALIZATION END------------------------------------------------#

#---------------------------------------------PARSING DATES START----------------------------------------------------#
    elif select_options=="Parsing Dates":
        st.subheader("Inspect the date column in your Dataset.")
        st.markdown("```Date``` column in your Dataset could contain some anomalies. Let's find out: ")
        original_data=dataframe.copy()
        date_columns=[i for i in original_data.columns if re.search("[d|D][a|A][t|T][e|E]"and "[t|T][i|I][m|M][e|E]", i)]
        if len(date_columns)!=0:
            select_date_column=st.selectbox(label="Inspect Date column:",options=date_columns)
            if select_date_column:
                datatype=original_data[select_date_column].dtype
                st.markdown(f"Data Type: ```{datatype}```")
                if datatype=='object':
                    st.markdown("Anomalies:")
                    date_lengths = original_data[select_date_column].str.len()
                    anomalies=date_lengths.value_counts()
                    st.dataframe(anomalies)
                    indices = np.where([date_lengths == max([i for i,j in anomalies.items()])])[1]
                    corrupted=pd.DataFrame(original_data.loc[indices]) #Shows the corrupted columns in dates
                    st.dataframe(corrupted.style.apply(lambda x: ['background-color: red' if i in indices else '' for i in x.index.values],subset=[select_date_column], axis=0))
                elif datatype=='datetime':
                    st.markdown(f'```{select_date_column}``` has datetime formatted entries.')
        else:
            st.markdown("There are no ```Date``` columns in your data")
#---------------------------------------------Parsing Dates END------------------------------------------------------#

#------------------------------------------INCONSISTENT DATA ENTRY---------------------------------------------------#
    elif select_options=="Inconsistent Data Entry":
        columns=[i for i in dataframe.select_dtypes('object')]
        select_options=st.sidebar.selectbox(label='Options:',options=['Unique Values','FuzzyWuzzy'])
        selected_column=st.selectbox(label="Columns:",options=columns)
        try:
            entries=dataframe[selected_column].unique()
                # entries=sorted(entries)
            if select_options=='Unique Values':
                entries=pd.DataFrame(entries,columns=["unique"])
                st.dataframe(entries,width=50000)
                st.write(f"Column: ```{selected_column}```")
                st.write(f"Unique values:\t```{entries.shape[0]}```")
            elif select_options=='FuzzyWuzzy':
                matcher=st.text_input(label="Write what you want to match:",placeholder="Enter match word...")
                limit=st.number_input(label="Choose Limit:",value=0)
                if (matcher!=None) and (limit!=0):
                    matches = fuzzywuzzy.process.extract(matcher, entries, limit=limit, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
                    # take a look at them
                    matches=pd.DataFrame(matches,columns=["matches","scores"])
                    st.dataframe(matches)
        except:
            st.error("There are no inconsistent entries")

#-----------------------------------------INCONSISTENT DATA ENTRY END---------------------------------------------------#

#-------------------------------------------FEATURE SELECTOINS START----------------------------------------------------#

    elif select_options=="Feature Selections":
        feature_select_options=st.sidebar.selectbox(label="Choose Options:",options=[None,"Mutual Informations","Correlations","Clusters","Chi2","Feature Importance","PCA"])
        X = dataframe.copy()    
        X = X.dropna(axis=0)
        columns=[i for i in X.columns if not re.search("[i|I][d|D]", i)]
        for colname in X.select_dtypes(["object","bool"]):
                X[colname], _ = X[colname].factorize()
        if feature_select_options==None:
            st.markdown("```None```")
        elif feature_select_options=="Mutual Informations":

            target=st.selectbox(label="Select Target:",options=[None,*columns])
            
            st.write(f"Target: ```{target}```")
            X=X.copy()
            if target!=None:
                y = X.pop(target)
                discrete_features = X.dtypes == int
                mi_scores = make_mi_scores(X, y, discrete_features)
                fig=plt.figure(dpi=100, figsize=(8, 5))
                plot_mi_scores(mi_scores)
                st.pyplot(fig)
            
        elif feature_select_options=="Correlations":
            palette=st.selectbox(label="Color Palettes:",options=["YlGnBu","plasma","viridis","rocket","cubehelix","rocket_r","crest","mako"])
            linewidth=st.number_input(label="Choose Linewidth",value=0)
            if linewidth>0:
                linecolor=st.selectbox(label="Choose Linecolor",options=[None,"black","white","red","green"])
            else:
                linecolor=None
            st.header("Correlations Heatmap")
            fig=plt.figure(figsize=(20,10))
            plt.title(f"Correlations Heatmap to")
            sns.heatmap(X.corr(),annot=True,linecolor=linecolor,linewidths=linewidth,cmap = palette)
            st.pyplot(fig)
            if st.button("See code"):
                if linecolor!=None:
                    code=f"""import seaborn as sns 
sns.heatmap(X.corr(),annot=True,linecolor='{linecolor}',linewidths={linewidth},cmap = '{palette}')"""

                else:
                    code=f"""import seaborn as sns 
sns.heatmap(X.corr(),annot=True,linecolor=None,linewidths={linewidth},cmap = '{palette}')"""
                st.code(code, language='python')
        elif feature_select_options=="Clusters":
            df=X.copy()
            choose_columns=st.multiselect(label="Choose columns to find clusters:",options=columns)
            number_of_clusters=st.number_input(label="Choose cluster numbers:",value=0)
            
            if choose_columns!=[] and number_of_clusters>=2:
                _X = df.loc[:,choose_columns]
                kmeans = KMeans(n_clusters=number_of_clusters)
                _X["Cluster"] = kmeans.fit_predict(_X)
                _X["Cluster"] = _X["Cluster"].astype("category")
                
                plottype=st.selectbox(label="Plot type",options=["resplot","boxplot"])

                if plottype=="resplot":
                    x=st.selectbox(label="Select X:",options=choose_columns)
                    y=st.selectbox(label="Select Y:",options=choose_columns)
                    fig=sns.relplot( x=x, y=y, hue="Cluster", data=_X, height=6)
                    st.pyplot(fig)
                elif plottype=="boxplot":
                    X=_X.copy()
                    x=st.selectbox(label="Choose X",options=columns)
                    X[x] = df[x]
                    fig=sns.catplot(x=x, y="Cluster", data=X, kind="boxen", height=6)
                    st.pyplot(fig)
            else:
                st.error("Colum not chosen or Cluster number is not decided")
        elif feature_select_options=="Chi2":
            df=X.copy()
            columns=list(df.columns)
            drop=st.multiselect(label="Drop Columns",options=columns)
            if drop!=[]:
                df=df.drop(columns=drop,axis=1)
                for d in drop:
                    st.write(f"Column ```{d}``` dropped")
            columns=list(df.columns)
            with st.form("my_form"):
                st.header("$Chi^2$ Evaluation")
                st.markdown("Formula: $\chi^2=\sum\\frac{(O_i-E_i)^2}{E_i}$")
                st.markdown("$\chi = chi$")
                st.markdown("$O_i=\\text{observed value}$")
                st.markdown("$E_i = \\text{expected value}$")
                target=st.selectbox(label="Choose Target:",options=[None,*columns])
                k=st.number_input(label="How many best features you wish to find?",value=0)
                # Every form must have a submit button.
                scores=st.checkbox(label="Show Scores Data Frame")
                submitted = st.form_submit_button("Find Features")
                if submitted:
                    X=df.copy()
                    y=X.pop(target)
                    bestfeatures = SelectKBest(score_func=chi2, k=k)
                    fit = bestfeatures.fit(X,y)
                    dfscores = pd.DataFrame(fit.scores_)
                    dfcolumns = pd.DataFrame(X.columns)
                    #concat two dataframes for better visualization 
                    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
                    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
                    if scores:
                        st.dataframe(featureScores.sort_values(by=['Score'],ascending=False))
                    fig=plt.figure(figsize=(10,5))
                    sns.barplot(x=featureScores['Score'],y=featureScores['Specs'],orient='h')
                    st.markdown("$\\textbf{Features with most importance are shown below:}$")
                    st.pyplot(fig)
        elif feature_select_options=="Feature Importance":
            df=X.copy()
            columns=list(df.columns)
            drop=st.multiselect(label="Drop Columns",options=columns)
            if drop!=[]:
                df=df.drop(columns=drop,axis=1)
                for d in drop:
                    st.write(f"Column ```{d}``` dropped")
            columns=list(df.columns)
            with st.form("my_form"):
                st.header("Feature Importance")
                target=st.selectbox(label="Choose Target:",options=[None,*columns])
                k=st.number_input(label="How many best features you wish to find?",value=0)
                # Every form must have a submit button.
                scores=st.checkbox(label="Show Scores Data Frame")
                submitted = st.form_submit_button("Find Features")
                if submitted:
                    X=df.copy()
                    y=X.pop(target)
                    model = ExtraTreesClassifier()
                    model.fit(X,y)
                    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
                    fig=plt.figure(figsize=(20,10))
                    feat_importances.nlargest(k).plot(kind='barh')
                    st.pyplot(fig)
        elif feature_select_options=="PCA":
            with st.form('PCA_form'):
                columns=[i for i in dataframe.select_dtypes(['float','bool'])]
                target=st.selectbox(label="Select a Target: ",options=[None,*columns])
                features = st.multiselect(label="Select features:",options=columns)
                submit = st.form_submit_button("Do PCA")
                if submit:
                    try:
                        X = dataframe.copy()
                        X = X.dropna()
                        y = X.pop(target)
                        X = X.loc[:, features]
                        # Standardize
                        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
                        # Create principal components
                        pca = PCA()
                        X_pca = pca.fit_transform(X_scaled)

                        # Convert to dataframe
                        component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
                        X_pca = pd.DataFrame(X_pca, columns=component_names)
                        st.dataframe(X_pca.head())
                        loadings = pd.DataFrame(
                        pca.components_.T,  # transpose the matrix of loadings
                        columns=component_names,  # so the columns are the principal components
                        index=X.columns,  # and the rows are the original features
                        )
                        st.dataframe(loadings)
                        st.subheader("Principal Components Variance")
                        plot_variance(pca)
                        st.subheader("Mutual Information scores")
                        mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
                        fig=plt.figure(dpi=100, figsize=(8, 5))
                        plot_mi_scores(mi_scores)
                        st.pyplot(fig)
                    except:
                        st.error('Something went wrong...')
                        
#--------------------------------------FEATURE SELECTIONS END-------------------------------------------------#

    elif select_options=="Select...":
        st.header("Start Inspection 🐼")
        st.write("Selected: ```None```")
        st.write(dataframe.head())
        types=dataframe.dtypes.astype(str)
        types=types.to_frame(name="types")
        st.header("Datatypes")
        st.dataframe(types)



####-------------------------------------FOOTER------------------------------------------####
hide_streamlit_style = """
            <head>
            <style>
            #MainMenu{visibility: hidden;}
            .css-fk4es0{display:none;}
            .css-1lsmgbg {display: none;}
            .myFooter{color:rgba(250, 250, 250, 0.6); margin-top: 150px;}
            .myFooter a{color: rgb(255, 75, 75); font-weight: bolder;}
            .css-17ziqus {background-color: brown; visibility: visible}
            </style>
            <title> Book Tracker </title>
            </head>
            <div class="myFooter">© 2022 Copyright | Made by <a href="https://codingwithzk.netlify.app" >Md. Ziaul Karim</a>| <a>ziaul.karim497@gmail.com</a></div>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 