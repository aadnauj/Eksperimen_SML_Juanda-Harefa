import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


# ============================================================
# FUNGSI 1: LOAD DATASET
# ============================================================
def load_data(file_id: str = None, filepath: str = None) -> pd.DataFrame:
    """
    Load dataset dari Google Drive (file_id) atau path lokal (filepath).

    Parameters:
        file_id  (str): ID file Google Drive
        filepath (str): Path file lokal (.csv)

    Returns:
        pd.DataFrame: Raw dataset
    """
    if file_id:
        url = f"https://drive.google.com/uc?id={file_id}"
        data = pd.read_csv(url)
        print(f"✅ Dataset berhasil dimuat dari Google Drive (ID: {file_id})")
    elif filepath:
        data = pd.read_csv(filepath)
        print(f"✅ Dataset berhasil dimuat dari: {filepath}")
    else:
        raise ValueError("Harap berikan file_id atau filepath!")

    print(f"   Shape awal: {data.shape}")
    return data


# ============================================================
# FUNGSI 2: DROP KOLOM TIDAK RELEVAN
# ============================================================
def drop_irrelevant_columns(data: pd.DataFrame,
                             columns: list = None) -> pd.DataFrame:
    """
    Menghapus kolom yang tidak relevan untuk training.

    Parameters:
        data    (pd.DataFrame): Dataset
        columns (list)        : Daftar kolom yang akan dihapus

    Returns:
        pd.DataFrame: Dataset tanpa kolom yang dihapus
    """
    if columns is None:
        columns = ['RowNumber', 'CustomerId', 'Surname']

    cols_exist = [c for c in columns if c in data.columns]
    data = data.drop(columns=cols_exist)
    print(f"✅ Kolom dihapus: {cols_exist}")
    print(f"   Shape setelah drop: {data.shape}")
    return data


# ============================================================
# FUNGSI 3: TANGANI MISSING VALUES
# ============================================================
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Mengecek dan menangani missing values.
    - Kolom numerik  → diisi dengan median
    - Kolom kategori → diisi dengan modus

    Parameters:
        data (pd.DataFrame): Dataset

    Returns:
        pd.DataFrame: Dataset tanpa missing values
    """
    total_missing = data.isnull().sum().sum()
    print(f"✅ Total missing values ditemukan: {total_missing}")

    if total_missing > 0:
        num_cols = data.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = data.select_dtypes(include=['object']).columns

        data[num_cols] = data[num_cols].fillna(data[num_cols].median())
        data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
        print("   Missing values berhasil ditangani (median & modus).")
    else:
        print("   Tidak ada missing values — data bersih!")

    return data


# ============================================================
# FUNGSI 4: HAPUS DATA DUPLIKAT
# ============================================================
def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Mendeteksi dan menghapus baris duplikat.

    Parameters:
        data (pd.DataFrame): Dataset

    Returns:
        pd.DataFrame: Dataset tanpa duplikat
    """
    before = data.shape[0]
    data = data.drop_duplicates()
    after = data.shape[0]
    print(f"✅ Data duplikat dihapus: {before - after} baris")
    print(f"   Shape setelah hapus duplikat: {data.shape}")
    return data


# ============================================================
# FUNGSI 5: DETEKSI & TANGANI OUTLIER (IQR Capping)
# ============================================================
def handle_outliers(data: pd.DataFrame,
                    target_col: str = 'Exited',
                    visualize: bool = False) -> pd.DataFrame:
    """
    Mendeteksi dan menangani outlier menggunakan metode IQR (capping/winsorizing).

    Parameters:
        data        (pd.DataFrame): Dataset
        target_col  (str)         : Kolom target yang dikecualikan
        visualize   (bool)        : Tampilkan boxplot jika True

    Returns:
        pd.DataFrame: Dataset dengan outlier yang sudah ditangani
    """
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    if visualize:
        fig, axes = plt.subplots(2, (len(num_cols) + 1) // 2, figsize=(16, 8))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            axes[i].boxplot(data[col].dropna())
            axes[i].set_title(f'Boxplot {col}')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Boxplot Sebelum Penanganan Outlier', fontsize=13)
        plt.tight_layout()
        plt.show()

    print("✅ Penanganan outlier (IQR Capping):")
    for col in num_cols:
        Q1  = data[col].quantile(0.25)
        Q3  = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower   = Q1 - 1.5 * IQR
        upper   = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower) | (data[col] > upper)].shape[0]
        data[col] = data[col].clip(lower=lower, upper=upper)
        print(f"   {col}: {outliers} outlier di-cap")

    return data


# ============================================================
# FUNGSI 6: ENCODING DATA KATEGORIKAL
# ============================================================
def encode_categorical(data: pd.DataFrame,
                        columns: list = None) -> pd.DataFrame:
    """
    Mengubah kolom kategorikal menjadi numerik menggunakan LabelEncoder.

    Parameters:
        data    (pd.DataFrame): Dataset
        columns (list)        : Kolom kategorikal yang akan di-encode

    Returns:
        pd.DataFrame: Dataset dengan kolom ter-encode
    """
    if columns is None:
        columns = ['Geography', 'Gender']

    le = LabelEncoder()
    cols_exist = [c for c in columns if c in data.columns]

    for col in cols_exist:
        data[col] = le.fit_transform(data[col])

    print(f"✅ Encoding selesai untuk kolom: {cols_exist}")
    return data


# ============================================================
# FUNGSI 7: BINNING
# ============================================================
def apply_binning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Mengelompokkan kolom Age dan Balance ke dalam kategori (binning).
    - Age    → Muda / Dewasa / Senior / Lansia
    - Balance → Zero / Low / Medium / High

    Parameters:
        data (pd.DataFrame): Dataset

    Returns:
        pd.DataFrame: Dataset dengan kolom baru Age_Group & Balance_Group
    """
    # Binning Age
    if 'Age' in data.columns:
        age_bins   = [0, 30, 45, 60, 100]
        age_labels = ['Muda', 'Dewasa', 'Senior', 'Lansia']
        data['Age_Group'] = pd.cut(data['Age'], bins=age_bins,
                                   labels=age_labels)
        # Encode Age_Group ke numerik
        age_order = {'Muda': 0, 'Dewasa': 1, 'Senior': 2, 'Lansia': 3}
        data['Age_Group'] = data['Age_Group'].map(age_order)
        print("✅ Binning Age selesai → kolom 'Age_Group' ditambahkan")

    # Binning Balance
    if 'Balance' in data.columns:
        bal_min    = data['Balance'].min() - 1
        bal_max    = data['Balance'].max() + 1
        bal_bins   = [bal_min, 0, 50000, 100000, bal_max]
        bal_labels = ['Zero', 'Low', 'Medium', 'High']
        data['Balance_Group'] = pd.cut(data['Balance'], bins=bal_bins,
                                        labels=bal_labels)
        bal_order = {'Zero': 0, 'Low': 1, 'Medium': 2, 'High': 3}
        data['Balance_Group'] = data['Balance_Group'].map(bal_order)
        print("✅ Binning Balance selesai → kolom 'Balance_Group' ditambahkan")

    return data


# ============================================================
# FUNGSI 8: NORMALISASI + SPLIT DATA
# ============================================================
def split_and_normalize(data: pd.DataFrame,
                         target_col: str = 'Exited',
                         test_size: float = 0.2,
                         random_state: int = 42):
    """
    Memisahkan fitur & target, lalu melakukan normalisasi MinMaxScaler
    dan split data train/test.

    Parameters:
        data         (pd.DataFrame): Dataset bersih
        target_col   (str)         : Nama kolom target
        test_size    (float)       : Proporsi data test (default 0.2)
        random_state (int)         : Random seed

    Returns:
        X_train, X_test, y_train, y_test (np.ndarray)
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler      = MinMaxScaler()
    num_cols    = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"✅ Normalisasi & Split selesai:")
    print(f"   X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"   X_test : {X_test.shape}  | y_test : {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ============================================================
# FUNGSI UTAMA: run_preprocessing()
# ============================================================
def run_preprocessing(file_id: str = None,
                       filepath: str = None,
                       target_col: str = 'Exited',
                       test_size: float = 0.2,
                       random_state: int = 42,
                       visualize_outlier: bool = False):
    """
    Fungsi utama yang menjalankan seluruh pipeline preprocessing
    secara otomatis dari load data hingga data siap dilatih.

    Parameters:
        file_id          (str)  : ID file Google Drive
        filepath         (str)  : Path file lokal
        target_col       (str)  : Nama kolom target
        test_size        (float): Proporsi data test
        random_state     (int)  : Random seed
        visualize_outlier(bool) : Tampilkan boxplot outlier

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("=" * 55)
    print("  AUTOMATE PREPROCESSING - Juanda Harefa")
    print("=" * 55)

    # Step 1: Load
    data = load_data(file_id=file_id, filepath=filepath)

    # Step 2: Drop kolom tidak relevan
    data = drop_irrelevant_columns(data)

    # Step 3: Missing values
    data = handle_missing_values(data)

    # Step 4: Hapus duplikat
    data = remove_duplicates(data)

    # Step 5: Outlier
    data = handle_outliers(data, target_col=target_col,
                           visualize=visualize_outlier)

    # Step 6: Encoding kategorikal
    data = encode_categorical(data)

    # Step 7: Binning
    data = apply_binning(data)

    # Step 8: Split & Normalisasi
    X_train, X_test, y_train, y_test = split_and_normalize(
        data,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state
    )

    print()
    print("=" * 55)
    print("  ✅ PREPROCESSING SELESAI! Data siap dilatih.")
    print("=" * 55)

    return X_train, X_test, y_train, y_test


# ============================================================
# MAIN — jalankan langsung sebagai script
# ============================================================
if __name__ == "__main__":
    # Ganti file_id dengan ID Google Drive dataset kamu
    FILE_ID = "19IfOP0QmCHccMu8A6B2fCUpFqZwCxuzO"

    X_train, X_test, y_train, y_test = run_preprocessing(
        file_id=FILE_ID,
        target_col='Exited',
        test_size=0.2,
        random_state=42,
        visualize_outlier=False   # Set True untuk lihat boxplot
    )

    print(f"\nX_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape  : {y_test.shape}")
