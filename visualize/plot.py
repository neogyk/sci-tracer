import plotly.graph_objs as go
from plotly.subplots import make_subplots

# After reducing embeddings to 3D (via PCA or t-SNE)
reduced_vectors = pca.fit_transform(sentence_embeddings)  # Shape: (n_sentences, 3)

# Create 3D scatter plot
scatter_plot = go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color='grey', opacity=0.5),
    text=[f"Sent {i}" for i in range(len(reduced_vectors))]
)

# Highlight query (index 0) and top results (indices 1–3)
highlighted_point = go.Scatter3d(
    x=[reduced_vectors[0, 0]],
    y=[reduced_vectors[0, 1]],
    z=[reduced_vectors[0, 2]],
    mode='markers',
    marker=dict(size=8, color='red', opacity=0.8),
    text=["Query"]
)

top_docs = go.Scatter3d(
    x=reduced_vectors[1:4, 0],
    y=reduced_vectors[1:4, 1],
    z=reduced_vectors[1:4, 2],
    mode='markers',
    marker=dict(size=8, color='blue', opacity=0.8),
    text=["Top 1", "Top 2", "Top 3"]
)

fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
fig.add_trace(scatter_plot)
fig.add_trace(highlighted_point)
fig.add_trace(top_docs)

fig.update_layout(
    scene=dict(
        xaxis=dict(title='PC1'),
        yaxis=dict(title='PC2'),
        zaxis=dict(title='PC3')
    ),
    title="3D Visualization of Sentence Embeddings with Trajectories"
)

fig.show()   