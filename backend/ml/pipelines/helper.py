import random
import re

# FEATURE PIPELINE HELPERS
def print_clean_df(df, num_rows=5, display_head=True, str_length=15):
    display_df = df.copy()
    rows = df.shape[0]
    cols = df.shape[1]

    # Get the columns that contain string data
    str_cols = display_df.select_dtypes(include=['object']).columns
    # Truncate string columns
    for col in str_cols:
        display_df[col] = display_df[col].astype(str).str.slice(0, str_length) + \
                          display_df[col].astype(str).str.len().gt(str_length).apply(lambda x: '...' if x else '')
    # Print the head or tail
    if display_head:
        print(display_df.head(num_rows).to_string())
        print(f"Number of rows: {rows}")
        print(f"Number of columns: {cols}")
    else:
        print(display_df.tail(num_rows).to_string())
        print(f"Number of rows: {rows}")
        print(f"Number of columns: {cols}")



# TRAINING PIPELINE HELPERS: