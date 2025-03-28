import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR  # Add SVR
from sklearn.impute import SimpleImputer  # For missing data handling
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix  # Add MAE
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        
        # Neural network configuration
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()
        
        
    def load_dataset(self):
        try:
            dataset_name = self.dataset_combo.currentText()
            if dataset_name == "Load Custom Dataset":
                return
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.load_boston()
            elif dataset_name == "California Housing Dataset":
                data = datasets.fetch_california_housing()
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(data.data, data.target, test_size=test_size, random_state=42)
            
            self.apply_scaling()
            self.apply_missing_data_handling()
            self.status_bar.showMessage(f"Loaded {dataset_name}")
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def load_custom_data(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "CSV files (*.csv)")
            if file_name:
                data = pd.read_csv(file_name)
                target_col = self.select_target_column(data.columns)
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
                    self.apply_scaling()
                    self.apply_missing_data_handling()
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
    def create_data_section(self):
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset", "Iris Dataset", "Breast Cancer Dataset",
            "Digits Dataset", "Boston Housing Dataset", "California Housing Dataset", "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["No Scaling", "Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
        
        # Add missing data handling dropdown
        self.missing_combo = QComboBox()
        self.missing_combo.addItems(["No Imputation", "Mean Imputation", "Interpolation", "Forward Fill", "Backward Fill"])
        
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Missing Data:"))
        data_layout.addWidget(self.missing_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)
        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)
    
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
        
        lr_group = self.create_algorithm_group("Linear Regression", {"fit_intercept": "checkbox"})
        regression_layout.addWidget(lr_group)
        
        rfr_group = self.create_algorithm_group("Random Forest Regressor", 
                                                {"n_estimators": "int", "max_depth": "int", "min_samples_split": "int"})
        regression_layout.addWidget(rfr_group)
        
        svr_group = self.create_algorithm_group("SVR", 
                                                {"C": "double", "epsilon": "double", "kernel": ["linear", "rbf", "poly"]})
        regression_layout.addWidget(svr_group)
        
        self.reg_loss_combo = QComboBox()
        self.reg_loss_combo.addItems(["MSE", "MAE", "Huber Loss"])
        regression_layout.addWidget(QLabel("Loss Function:"))
        regression_layout.addWidget(self.reg_loss_combo)
        
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        logistic_group = self.create_algorithm_group("Logistic Regression", 
                                                     {"C": "double", "max_iter": "int", "multi_class": ["ovr", "multinomial"]})
        classification_layout.addWidget(logistic_group)
        
        nb_group = self.create_algorithm_group("Naive Bayes", 
                                               {"var_smoothing": "double", "priors": "text"})
        classification_layout.addWidget(nb_group)
        
        svm_group = self.create_algorithm_group("Support Vector Machine", 
                                                {"C": "double", "kernel": ["linear", "rbf", "poly"], "degree": "int"})
        classification_layout.addWidget(svm_group)
        
        dt_group = self.create_algorithm_group("Decision Tree", 
                                               {"max_depth": "int", "min_samples_split": "int", "criterion": ["gini", "entropy"]})
        classification_layout.addWidget(dt_group)
        
        rf_group = self.create_algorithm_group("Random Forest Classifier", 
                                               {"n_estimators": "int", "max_depth": "int", "min_samples_split": "int"})
        classification_layout.addWidget(rf_group)
        
        knn_group = self.create_algorithm_group("K-Nearest Neighbors", 
                                                {"n_neighbors": "int", "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]})
        classification_layout.addWidget(knn_group)
        
        self.class_loss_combo = QComboBox()
        self.class_loss_combo.addItems(["Cross-Entropy", "Hinge Loss"])
        classification_layout.addWidget(QLabel("Loss Function:"))
        classification_layout.addWidget(self.class_loss_combo)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        return widget
    
    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        
        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)
        
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        
        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)
        
        return widget
    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_algorithm_group(self, name, params):
        group = QGroupBox(name)
        layout = QVBoxLayout()
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
                if param_name == "n_estimators": widget.setValue(100)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
                if param_name == "C": widget.setValue(1.0)
            elif param_type == "checkbox":
                widget = QCheckBox()
                widget.setChecked(True)
            elif param_type == "text":
                widget = QLineEdit()
                widget.setText("uniform")
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        group.setLayout(layout)
        return group

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
       
    def create_deep_learning_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        self.dl_loss_combo = QComboBox()
        self.dl_loss_combo.addItems(["Cross-Entropy", "Hinge Loss", "MSE", "MAE", "Huber Loss"])
        mlp_layout.addWidget(QLabel("Loss Function:"))
        mlp_layout.addWidget(self.dl_loss_combo)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    def train_model(self, model_name, param_widgets):
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first!")
            return
        try:
            params = {}
            for param_name, widget in param_widgets.items():
                if isinstance(widget, QSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QCheckBox):
                    params[param_name] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    if param_name == "priors" and widget.text() != "uniform":
                        params[param_name] = [float(x) for x in widget.text().split(",")]
                    else:
                        params[param_name] = None
            
            loss = self.reg_loss_combo.currentText() if "Regressor" in model_name or model_name == "Linear Regression" else self.class_loss_combo.currentText()
            
            if model_name == "Linear Regression":
                self.current_model = LinearRegression(**params)
            elif model_name == "Logistic Regression":
                self.current_model = LogisticRegression(**params)
            elif model_name == "Naive Bayes":
                self.current_model = GaussianNB(**params)
            elif model_name == "Support Vector Machine":
                self.current_model = SVC(**params)
            elif model_name == "SVR":
                self.current_model = SVR(**params)
            elif model_name == "Decision Tree":
                self.current_model = DecisionTreeClassifier(**params)
            elif model_name == "Random Forest Classifier":
                self.current_model = RandomForestClassifier(**params)
            elif model_name == "Random Forest Regressor":
                self.current_model = RandomForestRegressor(**params)
            elif model_name == "K-Nearest Neighbors":
                self.current_model = KNeighborsClassifier(**params)
            elif model_name == "K-Means Parameters":
                self.current_model = KMeans(**params)
                self.current_model.fit(self.X_train)
                y_pred = self.current_model.predict(self.X_test)
                self.update_visualization(y_pred)
                self.update_metrics(y_pred, loss)
                return
            
            self.current_model.fit(self.X_train, self.y_train)
            y_pred = self.current_model.predict(self.X_test)
            self.update_visualization(y_pred)
            self.update_metrics(y_pred, loss)
            self.status_bar.showMessage(f"Trained {model_name} successfully")
        except Exception as e:
            self.show_error(f"Error training {model_name}: {str(e)}")
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        group.setLayout(layout)
        return group
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def train_neural_network(self):
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        try:
            model = self.create_neural_network()
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            loss = self.dl_loss_combo.currentText().lower().replace(" ", "_")
            
            X_train = self.X_train.reshape(-1, self.X_train.shape[-1]) if len(self.X_train.shape) > 2 else self.X_train
            X_test = self.X_test.reshape(-1, self.X_test.shape[-1]) if len(self.X_test.shape) > 2 else self.X_test
            y_train = tf.keras.utils.to_categorical(self.y_train) if "cross" in loss or "hinge" in loss else self.y_train
            y_test = tf.keras.utils.to_categorical(self.y_test) if "cross" in loss or "hinge" in loss else self.y_test
            
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy' if "cross" in loss or "hinge" in loss else 'mae'])
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                                validation_data=(X_test, y_test), callbacks=[self.create_progress_callback()])
            self.plot_training_history(history)
            self.status_bar.showMessage("Neural Network Training Complete")
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
    
    def create_neural_network(self):
        model = models.Sequential()
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            if layer_type == "Dense":
                if len(model.layers) == 0:
                    params['input_shape'] = (self.X_train.shape[1],)
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
        
        num_classes = len(np.unique(self.y_train)) if "cross" in self.dl_loss_combo.currentText().lower() else 1
        model.add(layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'linear'))
        return model
   
                   
    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
                
            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)
                
        return ProgressCallback(self.progress_bar)
        
    
        
        
        
    def update_visualization(self, y_pred):
        self.figure.clear()
        if self.current_model.__class__.__name__.endswith('Regressor') or len(np.unique(self.y_test)) > 10:
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred, alpha=0.5)
            ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title(f"{self.current_model.__class__.__name__} Predictions")
        else:
            ax = self.figure.add_subplot(111)
            cm = confusion_matrix(self.y_test, y_pred)
            cax = ax.matshow(cm, cmap='Blues')
            self.figure.colorbar(cax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        self.canvas.draw()

    def update_metrics(self, y_pred, loss):
        metrics_text = "Model Performance Metrics:\n\n"
        if self.current_model.__class__.__name__.endswith('Regressor') or len(np.unique(self.y_test)) > 10:
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = self.current_model.score(self.X_test, self.y_test)
            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Mean Absolute Error: {mae:.4f}\n"
            if loss == "Huber Loss":
                metrics_text += "Huber Loss: Not directly supported by scikit-learn\n"
            metrics_text += f"R² Score: {r2:.4f}"
        else:
            accuracy = accuracy_score(self.y_test, y_pred)
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n" + str(confusion_matrix(self.y_test, y_pred))
        self.metrics_text.setText(metrics_text)
    
        
    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])
        
        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
        
    def apply_missing_data_handling(self):
        """Apply selected missing data handling method"""
        method = self.missing_combo.currentText()
        if method != "No Imputation" and self.X_train is not None:
            try:
                if method == "Mean Imputation":
                    imputer = SimpleImputer(strategy='mean')
                    self.X_train = imputer.fit_transform(self.X_train)
                    self.X_test = imputer.transform(self.X_test)
                elif method == "Interpolation":
                    self.X_train = pd.DataFrame(self.X_train).interpolate().fillna(method='bfill').to_numpy()
                    self.X_test = pd.DataFrame(self.X_test).interpolate().fillna(method='bfill').to_numpy()
                elif method == "Forward Fill":
                    self.X_train = pd.DataFrame(self.X_train).fillna(method='ffill').to_numpy()
                    self.X_test = pd.DataFrame(self.X_test).fillna(method='ffill').to_numpy()
                elif method == "Backward Fill":
                    self.X_train = pd.DataFrame(self.X_train).fillna(method='bfill').to_numpy()
                    self.X_test = pd.DataFrame(self.X_test).fillna(method='bfill').to_numpy()
            except Exception as e:
                self.show_error(f"Error handling missing data: {str(e)}")

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
