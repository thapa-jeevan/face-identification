from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.eigen_face.data_utils import load_images
from src.models.linear_discriminant_analysis import LDA
from src.models.pca import PCA
from src.utils import calculate_accuracy
from .data_utils import get_face_id_data
from src.utils import seed_everything

seed_everything(98123)

if __name__ == '__main__':
    pca = PCA()
    pca.fit(load_images())

    X_train_, X_test_, y_train, y_test = get_face_id_data()
    X_train = pca.transform(X_train_, k=200)
    X_test = pca.transform(X_test_, k=200)

    for model_C in [KNeighborsClassifier, SVC, RandomForestClassifier, LDA]:
        model = model_C()
        model.fit(X_train, y_train)

        train_acc = calculate_accuracy(model, X_train, y_train)
        test_acc = calculate_accuracy(model, X_test, y_test)
        print(f"{model}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")
