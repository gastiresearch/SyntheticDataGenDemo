import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from IPython.display import display, Image
from IPython.display import clear_output
import io
from sklearn.utils import resample

def add_missing_data_n(df, missing_percentage=0.20):
    total_values = df.size
    num_missing = int(total_values * missing_percentage)
    random_numbers = random.sample(range(0, 263), 100)
    all_positions = [(row, col) for row in range(df.shape[0]) for col in random_numbers]
    missing_positions = np.random.choice(len(all_positions), num_missing, replace=False)
    for pos in missing_positions:
        row, col = all_positions[pos]
        df.iat[row, col] = np.nan
    return df

def add_missing_data(df, percent_missing=0.20, random_seed=None):
    """
    Aggiunge dati mancanti (NaN) al percentuale specificata delle righe di un DataFrame.

    Parametri:
    - df (pd.DataFrame): Il DataFrame su cui aggiungere dati mancanti.
    - percent_missing (float): La percentuale di righe a cui aggiungere dati mancanti.
    - random_seed (int, optional): Il seme per la riproducibilità dei risultati casuali.

    Restituisce:
    - pd.DataFrame: Il DataFrame modificato con dati mancanti aggiunti.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    num_rows = len(df)
    num_missing_rows = int(num_rows * percent_missing)
    
    missing_indices = np.random.choice(df.index, num_missing_rows, replace=False)
    
    for idx in missing_indices:
        col_to_nan = np.random.choice(df.columns)
        df.at[idx, col_to_nan] = np.nan
    
    return df

def add_missing_data_to_20_percent_rows(df, missing_percentage=0.20):
    # Calcola il numero totale di righe nel DataFrame
    total_rows = df.shape[0]
    
    # Calcola il numero di righe che devono avere dati mancanti
    num_rows_with_missing = int(total_rows * missing_percentage)
    
    # Seleziona casualmente le righe da riempire con dati mancanti
    rows_to_nan = np.random.choice(df.index, size=num_rows_with_missing, replace=False)
    
    for row in rows_to_nan:
        # Seleziona casualmente quante e quali colonne devono avere dati mancanti in questa riga
        num_cols_with_missing = np.random.randint(1, df.shape[1] + 1)
        cols_to_nan = np.random.choice(df.columns, size=num_cols_with_missing, replace=False)
        
        # Imposta i valori selezionati a NaN
        df.loc[row, cols_to_nan] = np.nan
    
    return df

def plot_missing_data_histogram(df):
    missing_percent = df.isnull().mean() * 100
    plt.figure(figsize=(15, 8))
    missing_percent.plot(kind='bar', color='skyblue')
    plt.title('Percentuale di dati mancanti per colonna')
    plt.xlabel('Colonne')
    plt.ylabel('Percentuale di dati mancanti (%)')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_dataframe_rows_histogram(df1, df2):
    """
    Crea un istogramma che mostra il numero di righe in due DataFrame.

    Parametri:
    - df1 (pd.DataFrame): Il primo DataFrame.
    - df2 (pd.DataFrame): Il secondo DataFrame.
    """
    num_rows_df1 = len(df1)
    num_rows_df2 = len(df2)

    # Creazione dell'istogramma
    plt.figure(figsize=(8, 6))
    plt.bar(['DataFrame 1', 'DataFrame 2'], [num_rows_df1, num_rows_df2], color=['blue', 'orange'])
    plt.xlabel('DataFrame')
    plt.ylabel('Numero di righe')
    plt.title('Numero di righe nei DataFrame')
    plt.show()


def plot_dataset_shapes(X_train, X_test, Y_train, Y_test):
    """
    Plotta un istogramma della shape dei dataset di training e test.

    Parametri:
    - X_train: array o DataFrame, dati di training.
    - X_test: array o DataFrame, dati di test.
    - Y_train: array o DataFrame, etichette di training.
    - Y_test: array o DataFrame, etichette di test.
    """
    dataset_shapes = {
        'X_train': X_train.shape[0],
        'X_test': X_test.shape[0],
        'Y_train': Y_train.shape[0],
        'Y_test': Y_test.shape[0]
    }

    plt.figure(figsize=(8, 6))
    plt.bar(dataset_shapes.keys(), dataset_shapes.values(), color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Dataset')
    plt.ylabel('Numero di righe')
    plt.title('Shape dei dataset di training e test')
    plt.show()

class PlotAccuracy(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        plt.plot(self.acc, label='Train Accuracy')
        plt.plot(self.val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('accuracy_plot.png')
        plt.close()
        img = Image(filename="accuracy_plot.png")
        clear_output(wait=True)
        display(img)

def balance_classes(df, class_column):
    # Trova le classi e il numero massimo di campioni per classe
    class_counts = df[class_column].value_counts()
    max_samples = class_counts.max()
    
    # Campiona le classi per bilanciare
    df_balanced = pd.DataFrame()
    for cls in class_counts.index:
        df_class = df[df[class_column] == cls]
        df_class_balanced = resample(df_class, 
                                     replace=True,    # campionamento con sostituzione
                                     n_samples=max_samples, # numero di campioni massimi
                                     random_state=42) # per la riproducibilità
        df_balanced = pd.concat([df_balanced, df_class_balanced])
    
    return df_balanced

def merge_and_balance(df1, df2, class_column):
    # Bilancia le classi in entrambi i DataFrame
    df1_balanced = balance_classes(df1, class_column)
    df2_balanced = balance_classes(df2, class_column)
    
    # Unisci i DataFrame bilanciati
    df_merged = pd.concat([df1_balanced, df2_balanced]).reset_index(drop=True)
    
    # Ribalancia le classi nel DataFrame unito
    df_merged_balanced = balance_classes(df_merged, class_column)
    
    return df_merged_balanced