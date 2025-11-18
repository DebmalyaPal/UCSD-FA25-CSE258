# ---------------------------------------------------------------
# ----------------- RATING PREDICTION -----------------------------
# ---------------------------------------------------------------
def rating_prediction():
    """
    Rating prediction using Matrix Factorization with biases.
    - Loads training interactions and rating pairs
    - Trains MF model with early stopping
    - Performs hyperparameter search
    - Generates predictions for given user-book pairs
    """

    # ----------------- IMPORTS -----------------
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from itertools import product

    # ----------------- FILE PATHS -----------------
    directory_path = '../Datasets/assignment1/'
    train_ratings_file_path = directory_path + 'train_Interactions.csv.gz'
    ratings_pairs_file_path = directory_path + 'pairs_Rating.csv'
    prediction_ratings_file_path = directory_path + 'predictions_Rating.csv'

    # ----------------- LOAD DATA -----------------
    train_interactions_df = pd.read_csv(train_ratings_file_path)
    ratings_pair_df = pd.read_csv(ratings_pairs_file_path)

    # ----------------- USER/BOOK MAPPING -----------------
    # Map users and books to integer indices for matrix factorization
    user_map = {u: i for i, u in enumerate(train_interactions_df['userID'].unique())}
    book_map = {b: i for i, b in enumerate(train_interactions_df['bookID'].unique())}

    n_users = len(user_map)
    n_books = len(book_map)

    # Add mapped indices to dataframe
    train_interactions_df['u_idx'] = train_interactions_df['userID'].map(user_map)
    train_interactions_df['b_idx'] = train_interactions_df['bookID'].map(book_map)

    # ----------------- TRAIN/VALIDATION SPLIT -----------------
    # Currently using full dataset for both train and validation
    # (can be changed to actual split if needed)
    train_df, validation_df = train_interactions_df, train_interactions_df
    # train_df, validation_df = train_test_split(train_interactions_df, test_size=0.05, random_state=42)

    # ----------------- TRAINING FUNCTION -----------------
    def train_mf(train_df, val_df, k=1, lr=0.01, reg=0.05, max_epochs=15, patience=3):
        """
        Train Matrix Factorization model with biases and early stopping.
        Args:
            train_df: training dataframe
            val_df: validation dataframe
            k: latent dimension
            lr: learning rate
            reg: regularization coefficient
            max_epochs: maximum training epochs
            patience: early stopping patience
        Returns:
            best_mse: best validation MSE
            best_model: tuple (global_mean, bu, bb, P, Q)
        """

        # Initialize parameters
        global_mean = train_df['rating'].mean() - 0.02
        bu = np.zeros(n_users)  # user bias
        bb = np.zeros(n_books)  # book bias
        P = np.random.normal(0, 0.1, (n_users, k))  # user latent factors
        Q = np.random.normal(0, 0.1, (n_books, k))  # book latent factors

        best_mse = float("inf")
        best_model = None
        no_improve = 0

        # Training loop
        for epoch in range(max_epochs):
            # Shuffle training data
            train_df = train_df.sample(frac=1, random_state=epoch)

            # SGD updates
            for _, row in train_df.iterrows():
                u, b, r = row['u_idx'], row['b_idx'], row['rating']
                pred = global_mean + bu[u] + bb[b] + P[u, :] @ Q[b, :]
                err = r - pred

                # Update biases
                bu[u] += lr * (err - reg * bu[u])
                bb[b] += lr * (err - reg * bb[b])

                # Update latent factors
                P[u, :] += lr * (err * Q[b, :] - reg * P[u, :])
                Q[b, :] += lr * (err * P[u, :] - reg * Q[b, :])

            # Validation MSE
            se = []
            for _, row in val_df.iterrows():
                u, b, r = row['u_idx'], row['b_idx'], row['rating']
                pred = global_mean + bu[u] + bb[b] + P[u, :] @ Q[b, :]
                se.append((r - pred) ** 2)
            mse = np.mean(se)
            print(f"Epoch {epoch+1}: Validation MSE={mse:.4f}")

            # Early stopping check
            if mse < best_mse:
                best_mse = mse
                best_model = (global_mean, bu.copy(), bb.copy(), P.copy(), Q.copy())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered.")
                    break

        return best_mse, best_model

    # ----------------- HYPERPARAMETER GRID SEARCH -----------------
    param_grid = {
        "k": [1],       # latent dimension
        "lr": [0.01],   # learning rate
        "reg": [0.05],  # regularization
    }

    best_mse = float("inf")
    best_params = None
    best_model = None

    # Try all parameter combinations
    for k, lr, reg in product(param_grid["k"], param_grid["lr"], param_grid["reg"]):
        mse, model = train_mf(train_df, validation_df, k=k, lr=lr, reg=reg, max_epochs=15, patience=3)
        print(f"Params: k={k}, lr={lr}, reg={reg} -> MSE={mse:.4f}")
        if mse < best_mse:
            best_mse = mse
            best_params = (k, lr, reg)
            best_model = model

    # ----------------- PREDICTION FUNCTION -----------------
    global_mean, bu, bb, P, Q = best_model
    k = best_params[0]

    def predict(u, b):
        """
        Predict rating for user u and book b.
        Handles unseen users/books by falling back to global mean.
        """
        if u not in user_map and b not in book_map:
            return global_mean

        u_idx = user_map.get(u, None)
        b_idx = book_map.get(b, None)

        bu_val = bu[u_idx] if u_idx is not None else 0.0
        bb_val = bb[b_idx] if b_idx is not None else 0.0
        pu = P[u_idx, :] if u_idx is not None else np.zeros(k)
        qb = Q[b_idx, :] if b_idx is not None else np.zeros(k)

        return global_mean + bu_val + bb_val + pu @ qb

    # ----------------- GENERATE PREDICTIONS -----------------
    with open(prediction_ratings_file_path, "w") as out:
        for line in open(ratings_pairs_file_path):
            if line.startswith("userID"):
                out.write(line)  # write header
                continue
            u, b = line.strip().split(",")
            pred = predict(u, b)

            # Clip predictions to [0, 5]
            pred = max(0.0, min(5.0, pred))

            out.write(f"{u},{b},{pred}\n")

    print(f"\nPredictions written to {prediction_ratings_file_path}")


# ---------------------------------------------------------------
# ----------------- READ PREDICTION -----------------------------
# ---------------------------------------------------------------
def read_prediction():
    """
    Read prediction using classification models.
    - Loads user-book interaction data
    - Builds positive and negative pairs
    - Adds popularity-based features
    - Trains a classifier (Random Forest by default)
    - Predicts whether a user will read a book
    """

    # ----------------- IMPORTS -----------------
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    # ----------------- FILE PATHS -----------------
    directory_path = '../Datasets/assignment1/'
    train_interactions_file_path = directory_path + 'train_Interactions.csv.gz'
    read_pairs_file_path = directory_path + 'pairs_Read.csv'
    prediction_read_file_path = directory_path + 'predictions_Read.csv'

    # ----------------- LOAD DATA -----------------
    ratings = pd.read_csv(train_interactions_file_path)  # columns: userID, bookID, rating
    pairs_df = pd.read_csv(read_pairs_file_path)

    # ----------------- USER/BOOK INDEX MAPPING -----------------
    user2idx = {u: i for i, u in enumerate(ratings['userID'].unique())}
    book2idx = {b: i for i, b in enumerate(ratings['bookID'].unique())}

    # ----------------- USER DETAILS -----------------
    # Dictionary storing number of books read per user
    user_details = {
        user_id: {'books_read': group['rating'].size}
        for user_id, group in ratings.groupby('userID')
    }

    # ----------------- BOOK DETAILS -----------------
    # Dictionary storing rating count and average rating per book
    book_details = {
        book_id: {
            'rating_count': group['rating'].size,
            'avg_rating': group['rating'].mean()
        }
        for book_id, group in ratings.groupby('bookID')
    }

    # ----------------- GLOBAL AVERAGES -----------------
    average_books_read_per_user = np.mean([u['books_read'] for u in user_details.values()])
    average_rating_count_per_book = np.mean([b['rating_count'] for b in book_details.values()])
    average_avg_rating_per_book = np.mean([b['avg_rating'] for b in book_details.values()])

    # ----------------- BUILD POSITIVE PAIRS -----------------
    positive_pairs = ratings[['userID', 'bookID']].copy()
    positive_pairs['label'] = 1

    # ----------------- BUILD NEGATIVE PAIRS -----------------
    users = ratings['userID'].unique()
    books = ratings['bookID'].unique()

    # All possible user-book pairs
    all_pairs = pd.MultiIndex.from_product([users, books], names=['userID', 'bookID']).to_frame(index=False)

    # Remove positive pairs → keep only negatives
    negative_pairs = all_pairs.merge(
        positive_pairs[['userID', 'bookID']],
        on=['userID', 'bookID'],
        how='left',
        indicator=True
    )
    negative_pairs = negative_pairs[negative_pairs['_merge'] == 'left_only'].drop(columns=['_merge'])
    negative_pairs['label'] = 0

    # Balance dataset: sample negatives equal to positives
    negative_pairs = negative_pairs.sample(n=len(positive_pairs), random_state=42)

    # ----------------- COMBINE DATASET -----------------
    dataset = pd.concat([positive_pairs, negative_pairs], ignore_index=True)

    # ----------------- ADD FEATURES -----------------
    dataset['user_idx'] = dataset['userID'].map(user2idx)
    dataset['book_idx'] = dataset['bookID'].map(book2idx)

    dataset['user_books_read'] = dataset['userID'].map(lambda u: user_details[u]['books_read'])
    dataset['book_rating_count'] = dataset['bookID'].map(lambda b: book_details[b]['rating_count'])
    dataset['book_avg_rating'] = dataset['bookID'].map(lambda b: book_details[b]['avg_rating'])

    # ----------------- PREPARE PAIRS_DF -----------------
    pairs_df['user_idx'] = pairs_df['userID'].map(user2idx).fillna(-1).astype(int)
    pairs_df['book_idx'] = pairs_df['bookID'].map(book2idx).fillna(-1).astype(int)

    pairs_df['user_books_read'] = pairs_df['userID'].map(
        lambda u: user_details.get(u, {}).get("books_read", average_books_read_per_user)
    )
    pairs_df['book_rating_count'] = pairs_df['bookID'].map(
        lambda b: book_details.get(b, {}).get("rating_count", average_rating_count_per_book)
    )
    pairs_df['book_avg_rating'] = pairs_df['bookID'].map(
        lambda b: book_details.get(b, {}).get("avg_rating", average_avg_rating_per_book)
    )

    # ----------------- FEATURE SETS -----------------
    categorical_features = ['user_idx', 'book_idx']
    numeric_features = ['user_books_read', 'book_rating_count', 'book_avg_rating']

    # ----------------- PREPROCESSOR -----------------
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )

    # ----------------- MODELS -----------------
    # Logistic Regression (alternative option)
    log_reg = LogisticRegression(
        solver='saga',
        penalty='l2',
        max_iter=2000,
        C=2.0
    )
    # Random Forest (default choice)
    rg_reg = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    # ----------------- PIPELINE -----------------
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', rg_reg)  # using Random Forest here
    ])

    # ----------------- CROSS VALIDATION (optional) -----------------
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    # scores = cross_val_score(pipeline, dataset[categorical_features + numeric_features], dataset['label'], cv=cv, scoring='accuracy')
    # print("Cross-validation scores:", scores)
    # print("Mean accuracy:", np.mean(scores))

    # ----------------- TRAIN FINAL MODEL -----------------
    pipeline.fit(dataset[categorical_features + numeric_features], dataset['label'])

    # ----------------- PREDICT ON PAIRS -----------------
    X_predict = pairs_df[categorical_features + numeric_features]
    y_predict = pipeline.predict(X_predict)

    # ----------------- OUTPUT PREDICTIONS -----------------
    with open(prediction_read_file_path, "w") as out:
        out.write("userID,bookID,prediction\n")
        for (user_id, book_id), pred in zip(pairs_df[['userID', 'bookID']].values, y_predict):
            out.write(f"{user_id},{book_id},{pred}\n")

    print("Predictions written to", prediction_read_file_path)


# ---------------------------------------------------------------
# ----------------- CATEGORY PREDICTION -------------------------
# ---------------------------------------------------------------
def category_prediction():
    """
    Category prediction using text classification.
    - Loads training and test category data
    - Preprocesses text (lowercase, punctuation removal, stopwords, stemming, lemmatization)
    - Normalizes ratings per user
    - Adds log-transformed vote counts
    - Extracts TF-IDF features from reviews
    - Trains a classifier (Logistic Regression by default)
    - Predicts genre categories for test reviews
    """

    # ----------------- IMPORTS -----------------
    import gzip
    import numpy as np
    import pandas as pd
    import string
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack
    from sklearn.linear_model import LogisticRegression

    # ----------------- FILE PATHS -----------------
    directory_path = '../Datasets/assignment1/'
    train_category_file_path = directory_path + 'train_Category.json.gz'
    test_category_file_path = directory_path + 'test_Category.json.gz'
    category_pairs_file_path = directory_path + 'pairs_Category.csv'
    prediction_category_file_path = directory_path + 'predictions_Category.csv'

    # ----------------- LOAD TRAIN DATA -----------------
    file_json_data = []
    for json_data in gzip.open(train_category_file_path, "rt"):
        file_json_data.append(eval(json_data))
    train_category_df = pd.DataFrame(file_json_data)

    # ----------------- LOAD TEST DATA -----------------
    file_json_data = []
    for json_data in gzip.open(test_category_file_path, "rt"):
        file_json_data.append(eval(json_data))
    test_category_df = pd.DataFrame(file_json_data)

    # ----------------- CATEGORY DICTIONARY -----------------
    unique_genres_df = train_category_df[['genre', 'genreID']].drop_duplicates()
    category_dict = dict(zip(unique_genres_df['genre'], unique_genres_df['genreID']))

    # ----------------- TEXT PREPROCESSING -----------------
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    nlp = spacy.load("en_core_web_sm")

    def preprocess_text(text):
        """
        Preprocess a single review:
        1. Lowercase
        2. Remove punctuation
        3. Remove stopwords
        4. Apply stemming
        5. Apply lemmatization
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [w for w in words if w not in stop_words]
        stemmed = [stemmer.stem(w) for w in words]

        # Lemmatization using spaCy
        doc = nlp(" ".join(stemmed))
        lemmatized = [token.lemma_ for token in doc]

        return " ".join(lemmatized)

    # Apply preprocessing
    train_category_df['clean_review'] = train_category_df['review_text'].apply(preprocess_text)
    test_category_df['clean_review'] = test_category_df['review_text'].apply(preprocess_text)

    # ----------------- USER RATING NORMALIZATION -----------------
    user_stats = train_category_df.groupby('user_id')['rating'].agg(['min', 'max']).reset_index()
    user_dict = user_stats.set_index('user_id').to_dict(orient='index')

    def normalize_rating(row, user_dict):
        """
        Normalize rating per user to [-1, 1].
        Falls back to rating/5.0 if user has only one rating or is unseen.
        """
        user = row['user_id']
        rating = row['rating']
        if user in user_dict:
            r_min, r_max = user_dict[user]['min'], user_dict[user]['max']
            if r_max != r_min:
                return -1 + ((rating - r_min) / (r_max - r_min)) * 2
            else:
                return rating / 5.0
        else:
            return rating / 5.0

    train_category_df['rating_norm'] = train_category_df.apply(lambda row: normalize_rating(row, user_dict), axis=1)
    test_category_df['rating_norm'] = test_category_df.apply(lambda row: normalize_rating(row, user_dict), axis=1)

    # ----------------- VOTE LOG FEATURE -----------------
    train_category_df['votes_log'] = np.log1p(train_category_df['n_votes'])
    train_category_df['votes_log'].replace([np.inf, -np.inf], 0, inplace=True)
    train_category_df['votes_log'].fillna(0, inplace=True)

    test_category_df['votes_log'] = np.log1p(test_category_df['n_votes'])
    test_category_df['votes_log'].replace([np.inf, -np.inf], 0, inplace=True)
    test_category_df['votes_log'].fillna(0, inplace=True)

    # ----------------- TF-IDF FEATURE EXTRACTION -----------------
    tfidf = TfidfVectorizer(
        max_features=None,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=5,
        max_df=0.95,
        binary=False,
        stop_words='english'
    )

    X_train_tfidf = tfidf.fit_transform(train_category_df['clean_review'])
    X_test_tfidf = tfidf.transform(test_category_df['clean_review'])

    # Combine TF-IDF with numeric features
    X_train = hstack([X_train_tfidf, train_category_df[['rating_norm', 'votes_log']].values])
    X_test = hstack([X_test_tfidf, test_category_df[['rating_norm', 'votes_log']].values])

    y_train = train_category_df['genreID']

    # ----------------- CLASSIFIER -----------------
    clf = LogisticRegression(
        penalty='l2',
        C=3,
        solver='saga',
        max_iter=2500,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # ----------------- OUTPUT PREDICTIONS -----------------
    with open(prediction_category_file_path, 'w') as predictions_category_file:
        predictions_category_file.write("userID,reviewID,prediction\n")
        for index, row in test_category_df.iterrows():
            predictions_category_file.write(f"{row['user_id']},{row['review_id']},{y_pred[index]}\n")

    print("Predictions written to", prediction_category_file_path)


# ---------------------------------------------------------------
# ----------------- SCRIPT ENTRY POINT --------------------------
# ---------------------------------------------------------------
if __name__ == '__main__':
    # When this file is run directly (not imported as a module),
    # the following functions will execute in sequence:

    rating_prediction()     # Generate rating predictions and save to file
    read_prediction()       # Generate read predictions and save to file
    category_prediction()   # Generate category predictions and save to file
    
