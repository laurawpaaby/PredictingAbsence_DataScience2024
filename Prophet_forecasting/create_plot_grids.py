### GRID PLOTS ###
# This script created grid plots for displaying forecasting results and component plots

#### IMPORT PACKAGES ####
from PIL import Image
import matplotlib.image as mpimg

from helper_functions_forecasting import get_components, plot_components

# List of selected groups for forecasting
name_list = ['akutafdelingen_Adm.personale (8M_03)',
             'akutafdelingen_Lægepersonale (8M_01)',
             'akutafdelingen_Plejepersonale (8M_02)']

# Create plot for each group
for name in name_list:
    
    forecast, model = get_components(model_name = name, periods = 30, freq = 'D')
    plot_components(forecast, model).savefig(f'./Prophet_forecasting/forecasting_plots/plot_components_{name}.png')

    import matplotlib.pyplot as plt
    from PIL import Image

    # Load images
    image1 = Image.open(f"./Prophet_forecasting/forecasting_plots/subset_{name}.png")
    image2 = Image.open(f'./Prophet_forecasting/forecasting_plots/plot_components_{name}.png')

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Display forecast from 2024-
    axs[0].imshow(image1)
    axs[0].axis('off')

    # Display components
    axs[1].imshow(image2)
    axs[1].axis('off')

    # Make room for title
    fig.subplots_adjust(top=0.90)

    # Position
    axs[0].set_position([-0.03, -0.25, 0.7, 1.6])
    axs[1].set_position([0.55, 0.12, 0.45, 0.8]) 

    # Title
    fig.suptitle(f'Prediction using Prophet: {name}', fontsize=18, fontfamily='serif', fontweight='bold')

    # Save figure
    fig.savefig(f'./Prophet_forecasting/forecasting_plots/plot_grid_{name}.png')

