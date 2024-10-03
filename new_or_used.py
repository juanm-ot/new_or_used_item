import  json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import functions as fn

# Logger set up
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_logging():
    """Configures the logging settings for the script."""
    logging.info("Logger configured.")

# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    ##data = [json.loads(x) for x in open("MLA_100k.jsonlines")]
    ##target = lambda x: x.get("condition")
    ##N = -10000
    ##X_train = data[:N]
    ##X_test = data[N:]
    ##y_train = [target(x) for x in X_train]
    ##y_test = [target(x) for x in X_test]
    ##for x in X_test:
    ##    del x["condition"]
    ##return X_train, y_train, X_test, y_test
    try:
        logging.info("Loading dataset from 'MLA_100k.jsonlines'...")
        with open("data/raw/MLA_100k.jsonlines", 'r') as file:
            data = [json.loads(line) for line in file]
        df = pd.json_normalize(data)

        # Goal: Extract main dictionaries from jsonlines
        columns_to_keep_nested = ['seller_address', 'location', 'seller_contact', 'geolocation']
        processed_data = []

        for record in data:
            processed_record = {}
            for key, value in record.items():

                if key in columns_to_keep_nested:
                    processed_record[key] = value 
                else:
                    processed_record[key] = value
            processed_data.append(processed_record)

        df =  pd.DataFrame(processed_data)
        logging.info("Dataset loaded and processed successfully.%d rows loaded", len(df))
        logging.info("Preparing model dataframe...")
        
        # Process Dataframe columns
        dtf = df.copy()
        columns_with_empty_list = ['deal_ids', 'variations', 'attributes', 'descriptions', 'pictures']
        dtf = fn.process_columns_with_empty_lists(dtf, columns_with_empty_list)

        columns_to_boolean = ['official_store_id', 'parent_item_id', 'video_id', 'original_price', 'seller_contact']
        dtf = fn.create_boolean_columns(dtf, columns_to_boolean)

        ## One to one set up specific columns 
        ### accepts_mercadopago
        dtf['accepts_mercadopago'] = dtf['accepts_mercadopago'].astype(int)
        
        ### automatic_relist
        dtf['have_automatic_relist'] = dtf['automatic_relist'].astype(int)

        ### category_id 
        dtf['category_id'] = dtf['category_id'].str.strip()

        ### shipping
        dtf['local_pick_up'] = dtf['shipping'].apply(lambda x: x.get('local_pick_up'))
        dtf['local_pick_up'] = dtf['local_pick_up'].astype(int)
        dtf['free_shipping'] = dtf['shipping'].apply(lambda x: x.get('free_shipping'))
        dtf['free_shipping'] = dtf['free_shipping'].astype(int)

        ### location
        dtf['location'] = dtf['location'].astype(str)
        dtf['location'] = dtf['location'].replace('{}', np.nan)
        dtf['have_location'] = dtf['location'].notnull().astype(int)

        ### sub_status
        dtf['sub_status'] = dtf['sub_status'].astype(str)
        dtf['sub_status'] = dtf['sub_status'].str.strip()
        dtf['sub_status'] = dtf['sub_status'].replace('[]', 'no_status')
        dtf['sub_status'] = dtf['sub_status'].str.replace(r"[\[\]']", '', regex=True)

        ### tags
        dtf['tags'] = dtf['tags'].apply(lambda x: x if isinstance(x, list) else [])
        tags_dummies = dtf['tags'].apply(lambda x: pd.Series(1, index=x)).fillna(0).astype(int)
        dtf = pd.concat([dtf, tags_dummies], axis=1)

        # Data  to model
        columns_to_model = ['id', 'condition','category_id','sub_status', 'have_seller_contact', 'have_deal_ids' ,'free_shipping', 'local_pick_up', 
                        'have_variations', 'have_location', 'listing_type_id', 'have_attributes', 'buying_mode', 'dragged_bids_and_visits', 
                        'good_quality_thumbnail', 'dragged_visits', 'free_relist', 'poor_quality_thumbnail', 'have_parent_item_id', 'have_descriptions', 
                        'have_pictures', 'have_official_store_id', 'accepts_mercadopago', 'have_original_price', 'have_automatic_relist', 'have_video_id',
                        'initial_quantity', 'sold_quantity', 'available_quantity', 'base_price']

        df_to_model = dtf[columns_to_model]
        categorical_columns = ['sub_status', 'listing_type_id', 'buying_mode', 'category_id']
        df_onehot = pd.get_dummies(df_to_model[categorical_columns])
        
        df_to_model_encoded = pd.concat([df_to_model, df_onehot], axis=1)
        df_to_model_encoded.drop(categorical_columns, axis=1, inplace=True)
        df_to_model_encoded['target_condition'] = df_to_model_encoded['condition'].apply(lambda x: 1 if x == 'new' else 0)

        logging.info("Model dataframe succesfully created")
        logging.info("Final Dataframe columns:")
        print(df_to_model_encoded.columns)

        dfm = df_to_model_encoded.copy()
        dfm = dfm.drop('condition', axis=1)

        X = dfm.drop(columns=['id', 'target_condition'])  
        y = dfm['target_condition']

        return train_test_split(X, y, test_size=10000, random_state=42)
    except Exception as e:
        logging.error(f"Error in build_dataset: {e}")
        raise

def train_and_evaluate_model(X_train, X_val, y_train, y_val):
    try:
        model_catboost = CatBoostClassifier(random_state=42, silent=True)  
        model_catboost.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100, plot=True)

        # Train evaluation metrics
        y_train_pred = model_catboost.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)

        # Test evaluation metrics
        y_val_pred = model_catboost.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)

        logging.info(f"Model metrics:")
        
        print(f"Accuracy en entrenamiento: {train_accuracy:.4f}")
        print(f"F1-Score en entrenamiento: {train_f1:.4f}")
        print(f"Accuracy en validación: {val_accuracy:.4f}")
        print(f"F1-Score en validación: {val_f1:.4f}")  

        # Save model
        logging.info(f"Exporting model...")
        model_catboost.save_model("new_or_used_catboost_model.cbm")
        logging.info(f"Model exported succesfully")

        # Return predictions and metrics
        metrics = {
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        }

        return y_val_pred, metrics
    except Exception as e:
        logging.error(f"Error in train_and_evaluate_model: {e}")
        raise

def plot_metrics(metrics, y_val, y_pred_catboost):
    try:
        conf_matrix = confusion_matrix(y_val, y_pred_catboost)
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion Matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Used', 'New'], yticklabels=['Used', 'New'], ax=axs[0])
        axs[0].set_title('Confusion Matrix')
        axs[0].set_xlabel('Predicted Labels')
        axs[0].set_ylabel('True Labels')

        # Overall Metrics
        bars = axs[1].bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        for bar in bars:
            yval = bar.get_height()
            axs[1].text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

        axs[1].set_title('Overall Metrics: Accuracy, Precision, Recall, F1-Score')
        axs[1].set_ylabel('Score')
        axs[1].set_ylim(0, 1)  

        plt.tight_layout()
        plt.savefig('confusion_matrix_and_metrics.png', format='png')
        plt.show()

        logging.info("Metrics plotted successfully.")
    except Exception as e:
        logging.error(f"Error in plot_metrics: {e}")
        raise

if __name__ == "__main__":
    configure_logging()
    try:
            logging.info("Starting the script...")
            X_train, X_val, y_train, y_val = build_dataset()
            y_pred_catboost, metrics = train_and_evaluate_model(X_train, X_val, y_train, y_val)
            plot_metrics(metrics, y_val, y_pred_catboost)

            
            logging.info("Script completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred in the main process: {e}")