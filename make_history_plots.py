
import numpy as np, matplotlib.pyplot as plt, pickle, glob, re

foldFolders = sorted(glob.glob('Fold*'))

fig1, ax1 = plt.subplots(figsize=(10,5))
fig2, ax2 = plt.subplots(figsize=(10,5))
fig3, ax3 = plt.subplots(figsize=(10,5))
fig4, ax4 = plt.subplots(figsize=(10,5))
fig5, ax5 = plt.subplots(figsize=(10,5))
fig6, ax6 = plt.subplots(figsize=(10,5))

def smoothen(data, kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(data, kernel, mode='valid')
    return data_convolved

smooth_factor = 5
linewidth = 2

for folder in foldFolders:
    foldnumber = re.findall(r'Fold_(\d+)', folder)
    assert len(foldnumber) == 1
    foldnumber = foldnumber[0]
    with open(folder + '/history_fold_'+foldnumber+'.pickle', 'rb') as f:
        history = pickle.load(f)

    toplot_ax1 = smoothen(history['val_loss'],smooth_factor)
    ax1.plot(toplot_ax1, label=folder, linewidth=linewidth)
    ax1.set_title('Validation Loss')
    ax1.set_ylabel('Sparse Categorical Crossentropy Loss')
    ax1.set_xlabel('Training Epoch')
    fig1.legend(bbox_to_anchor=(0.3, 0.8), fancybox=True, shadow=True)

    toplot_ax2 = smoothen(history['val_accuracy'],smooth_factor)
    ax2.plot(toplot_ax2, label=folder, linewidth=linewidth)
    ax2.set_title('Validation Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Training Epoch')
    fig2.legend(bbox_to_anchor=(0.3, 0.5), fancybox=True, shadow=True)

    toplot_ax3 = history['loss']
    ax3.plot(toplot_ax3, label=folder, linewidth=linewidth)
    ax3.set_title('Training Loss')
    ax3.set_ylabel('Sparse Categorical Crossentropy Loss')
    ax3.set_xlabel('Training Epoch')
    fig3.legend(bbox_to_anchor=(0.3, 0.8), fancybox=True, shadow=True)

    toplot_ax4 = history['accuracy']
    ax4.plot(toplot_ax4, label=folder, linewidth=linewidth)
    ax4.set_title('Training Accuracy')
    ax4.set_ylabel('Accuracy')
    ax4.set_xlabel('Training Epoch')
    fig4.legend(bbox_to_anchor=(0.3, 0.5), fancybox=True, shadow=True)

    # Combined Trainig and Validation Loss
    toplot_ax5 = smoothen(history['loss'],smooth_factor)
    ax5.plot(toplot_ax5, label=folder + ' Training', linewidth=linewidth)
    toplot_ax5 = smoothen(history['val_loss'],smooth_factor)
    ax5.plot(toplot_ax5, label=folder + ' Validation', linewidth=linewidth)
    ax5.set_title('Training and Validation Loss')
    ax5.set_ylabel('Sparse Categorical Crossentropy Loss')
    ax5.set_xlabel('Training Epoch')
    fig5.legend(bbox_to_anchor=(0.3, 0.8), fancybox=True, shadow=True)

    # Combined Trainig and Validation Accuracy
    toplot_ax6 = smoothen(history['accuracy'],smooth_factor)
    ax6.plot(toplot_ax6, label=folder + ' Training', linewidth=linewidth)
    toplot_ax6 = smoothen(history['val_accuracy'],smooth_factor)
    ax6.plot(toplot_ax6, label=folder + ' Validation', linewidth=linewidth)
    ax6.set_title('Training and Validation Accuracy')
    ax6.set_ylabel('Accuracy')
    ax6.set_xlabel('Training Epoch')
    fig6.legend(bbox_to_anchor=(0.3, 0.5), fancybox=True, shadow=True)


fig1.savefig('Validation_Loss.png')
fig2.savefig('Validation_Accuracy.png')
fig3.savefig('Training_Loss.png')
fig4.savefig('Training_Accuracy.png')
fig5.savefig('Training_and_Validation_Loss.png')
fig6.savefig('Training_and_Validation_Accuracy.png')
plt.close('all')
