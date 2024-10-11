import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import itertools

def load_vlad_data_and_labels(root_folder):
    """Load VLAD data and corresponding labels."""
    data, labels, class_names = [], [], []
    for action_class in os.listdir(root_folder):
        class_folder = os.path.join(root_folder, action_class)
        
        # Skip if not a directory
        if not os.path.isdir(class_folder):
            continue
        
        class_names.append(action_class)
        for file_name in os.listdir(class_folder):
            if file_name.endswith('.npy'):
                vlad_vector = np.load(os.path.join(class_folder, file_name))
                data.append(vlad_vector)
                labels.append(len(class_names) - 1)  # Assign based on current class_names length
                
    return np.array(data), np.array(labels), class_names

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, save_path='confusion_matrix.png'):
    """This function prints, plots, and saves the confusion matrix."""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the figure before showing
    plt.savefig(save_path, format='png', bbox_inches='tight')
    
    plt.show()

def main():
    root_folder = "/Users/rakhatm/Desktop/CV_Project/vlad_representation"
    
    # Load VLAD data and corresponding labels
    X, y, class_names = load_vlad_data_and_labels(root_folder)

    # Initialize classifier
    classifier = SVC(kernel='linear')
    
    # LOO cross-validation
    loo = LeaveOneOut()
    y_pred_all = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_pred_all.append(y_pred[0])

    # Calculate and print average accuracy
    avg_accuracy = accuracy_score(y, y_pred_all)
    print(f"Overall average accuracy using Leave-One-Out: {avg_accuracy * 100:.2f}%")

    # Calculate the confusion matrix and plot it
    cm = confusion_matrix(y, y_pred_all)
    plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png")

    # Compute accuracy for each class
    row_sums = np.sum(cm, axis=1)
    class_accuracies = cm.diagonal() / row_sums
    for class_name, accuracy in zip(class_names, class_accuracies):
        print(f"Accuracy for {class_name}: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
