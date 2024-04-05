import mlflow
import numpy as np
import streamlit as st
import plotly.express as px
from streamlit_extras.dataframe_explorer import dataframe_explorer

mlflow.set_tracking_uri(
    'https://dagshub.com/kynnemall/platelet-generator.mlflow'
)
st.set_page_config(
    page_title="Platelet Image Generator Comparison", layout="wide"
)


@st.cache_data
def load_run_data():
    """
    Load run data from the mlflow server on Dagshub

    Returns
    -------
    Pandas DataFrame
        Formatted dataframe of run data (parameters, metrics, etc.)

    """
    runs = mlflow.search_runs()
    runs.dropna(how='all', axis=1, inplace=True)

    # combine rows where a second run was initiated during the training loop
    as_above = runs[runs['tags.dagshub.labels.as_above'] == '-'].values
    as_below = runs[runs['tags.dagshub.labels.as_below'] == '-'].values
    whole_runs = runs[(runs['tags.dagshub.labels.as_above'].isna()) & (
        runs['tags.dagshub.labels.as_below'].isna())]

    for row_a, row_b in zip(as_above, as_below):
        new_row = []
        for item_a, item_b in zip(row_a, row_b):
            if item_a == item_b:
                new_row.append(item_a)
            elif item_a in (None, 'None', np.nan):
                new_row.append(item_b)
            elif item_b in (None, 'None', np.nan):
                new_row.append(item_a)
            else:
                new_row.append(item_b)
        whole_runs.loc[whole_runs.shape[0]] = new_row

    keep_cols = sorted(
        [c for c in whole_runs.columns if whole_runs[c].nunique() > 1]
    )
    whole_runs = whole_runs[keep_cols]

    return whole_runs


@st.cache_data
def load_images(artifact_uri, key):
    """
    Load generated images from the mlflow server

    Parameters
    ----------
    artifact_uri : string
        Mlflow uri to the artifact storage
    key : string
        Name of the sampler used to generate the images

    Returns
    -------
    images : dictionary
        Each key maps to the name of the method used to generate the images,
        which are stored in a list as 2D numpy arrays

    """
    images = []
    for n in range(1, 11):
        image_path = f'{artifact_uri}/{key}_Image{n:02}.png'
        pil_img = mlflow.artifacts.load_image(image_path)
        images.append(np.array(pil_img))
    return images


df = load_run_data()
fdf = dataframe_explorer(df)
st.dataframe(fdf)

with st.form('Filter runs and display images'):
    key = st.selectbox(
        'Choose which generated image format to view', ('Normal', 'GMM10')
    )
    submitted = st.form_submit_button('Submit')
    if submitted:
        prog_bar = st.progress(0, text='Loading Image Data')

        # load relevant images and combine into a (2, 5) tile view
        image_arrs = []
        for i, uri in enumerate(fdf['artifact_uri'], 1):
            images = load_images(uri, key)
            images = np.vstack([np.hstack(images[:5]), np.hstack(images[5:])])
            image_arrs.append(images)
            perc = i / fdf.shape[0]
            prog_bar.progress(perc)

        col1, col2 = st.columns(2)
        for i, (image_arr, idx) in enumerate(zip(image_arrs, fdf.index), 1):
            fig = px.imshow(image_arr, zmax=255)
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=0, b=0))
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            col = col1 if i % 2 else col2

            # get run info to title each set of images
            run = fdf.loc[idx]
            cfg = run['params.name'].replace('Config', '').replace('_', ' ')
            enc = run['params.encoder']
            dec = run['params.decoder']
            lat = run['params.latent_dim']
            psnr = run['metrics.PSNR']
            title = f'{cfg}, {enc}-{dec}, {lat} latents, PSNR: {psnr:.2f}'
            col.markdown(title)
            col.plotly_chart(fig, use_container_width=True)
