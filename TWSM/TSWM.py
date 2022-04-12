# compute term frequencies for text
def tf_from_string(text):
    
    (unique, counts) = np.unique(text.split(), return_counts=True)
    df_words = pd.DataFrame(unique, counts).reset_index().rename(columns = {"index":"counts", 0:"word"}).sort_values(by = "counts", ascending = False)
    df_words["rank"] =  df_words["counts"].rank(ascending=False)
    
    return df_words
