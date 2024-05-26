import pandas as pd

# Crear un DataFrame de ejemplo
data = {
    'fruta': ['manzana', 'manzana', 'naranja', 'naranja', 'manzana', 'naranja', 'manzana', 'naranja'],
    'cliente': ['Juan', 'Ana', 'Juan', 'Ana', 'Ana', 'Juan', 'Juan', 'Ana'],
    'ventas': [4, 5, 3, 3, 6, 2, 7, 2]
}

df = pd.DataFrame(data)

# Crear la tabla pivotante
pivot_df = df.pivot_table(index='cliente', columns='fruta', values='ventas', aggfunc='sum')

print(pivot_df)