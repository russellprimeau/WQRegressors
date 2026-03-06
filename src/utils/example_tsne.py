"""
Example usage of prepare_tsne_plot() function from preprocessing.py
"""
from pathlib import Path
from utils.preprocessing import prepare_tsne_plot

# Example 1: Load from CSV with all columns (except timestamp)
csv_path = Path("../data/input/sensors/Eurofins.csv")
embedding_df, fig = prepare_tsne_plot(csv_path)

print(f"Generated embedding with {len(embedding_df)} samples")
print(embedding_df.head())

# Display the interactive plot
fig.show()
""" 
# Example 2: Use only specific columns
specific_cols = ['01-Farge', '04-Turbiditet', '06-E.coli',
    '07-Intestinale enterokokker', '08-Kimtall 22°C', '09-Koliforme bakterier 37°C', '21-Arsen', '24-Bly',
    '32-Kadmium', '36-Kopper filtrert', '37-Krom', '41-Nikkel', 'Sink (Zn)']
embedding_df2, fig2 = prepare_tsne_plot(
    csv_path,
    feature_columns=specific_cols,
    perplexity=15,  # Adjust for smaller dataset
    scale=True
)

print(f"\nGenerated embedding with specific columns: {len(embedding_df2)} samples")
fig2.show()

# Example 3: Save embedding without displaying plot
embedding_df3, _ = prepare_tsne_plot(
    csv_path,
    feature_columns=['01-Farge', '04-Turbiditet'],
    return_fig=False
)

# Save the embedding coordinates
output_path = Path("../data/output/tsne_embedding.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
embedding_df3.to_csv(output_path, index=False)
print(f"\nSaved embedding to {output_path}")
 """