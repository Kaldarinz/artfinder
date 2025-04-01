import pandas as pd

# Example dictionary
data = {
    "column1": [1, 2, 3],
    "column2": ["a", "b", "c"],
    "column3": [1.1, 2.2, 3.3]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set individual column data types
df = df.astype({
    "column1": "int32",
    "column2": "string",
})

print(df.dtypes)