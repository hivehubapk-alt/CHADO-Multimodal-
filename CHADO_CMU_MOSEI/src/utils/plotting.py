import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_train_val_acc(train_acc, val_acc, out_path):
    plt.figure()
    plt.plot(train_acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(cm, out_path):
    plt.figure()
    plt.imshow(cm)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
