import sys
import os
import numpy as np
import scipy.io as sio
import h5py
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QTextEdit, QFileDialog, QFrame, QMessageBox,
                             QDialog, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt

# ==========================================
# 1. THE MODERN "UBUNTU/GEMINI" STYLESHEET
# ==========================================
STYLESHEET = """
/* Global Window Settings */
QMainWindow, QDialog {
    background-color: #282a36; 
    color: #f8f8f2;
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
}

/* Input Fields */
QLineEdit, QComboBox, QTextEdit {
    background-color: #343746; 
    border: 2px solid #44475a; 
    border-radius: 8px; 
    padding: 6px 12px; 
    color: #f8f8f2;
    selection-background-color: #bd93f9; 
}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
    border: 2px solid #bd93f9; /* Purple Focus */
    background-color: #3b3e4f; 
}

/* Buttons */
QPushButton {
    background-color: #6272a4; /* Ubuntu Purple */
    color: #ffffff;
    border: none;
    border-radius: 8px; 
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #bd93f9; /* Lighter Purple */
}
QPushButton:pressed {
    background-color: #4f5b85;
}

/* Cards/Frames */
QFrame#ContentFrame {
    background-color: #2d303e;
    border-radius: 12px;
    border: 1px solid #44475a;
}

/* Labels */
QLabel {
    color: #f8f8f2;
    font-weight: 500;
}
"""


# ==========================================
# 2. LOGIC & UI CLASS
# ==========================================

class LayerSelectionDialog(QDialog):
    """Popup Dialog for selecting layers"""

    def __init__(self, parent, source_data, dest_data):
        super().__init__(parent)
        self.setWindowTitle("Configure Layers")
        self.setFixedWidth(400)

        layout = QVBoxLayout()

        # --- Source Selection ---
        layout.addWidget(QLabel("Select Source Layer (From File 2):"))
        self.combo_source = QComboBox()
        if source_data.ndim == 2:
            self.combo_source.addItem("Layer 1 (2D)")
        else:
            for i in range(source_data.shape[2]):
                self.combo_source.addItem(f"Layer {i + 1}")
        layout.addWidget(self.combo_source)

        # --- Destination Selection ---
        layout.addWidget(QLabel("Select Destination (In File 1):"))
        self.combo_dest = QComboBox()
        self.combo_dest.addItem("Add as new layer")
        if dest_data.ndim == 2:
            self.combo_dest.addItem("Layer 1 (2D)")
        else:
            for i in range(dest_data.shape[2]):
                self.combo_dest.addItem(f"Layer {i + 1}")
        layout.addWidget(self.combo_dest)

        # --- Transpose Checkbox ---
        self.check_transpose = QCheckBox("Transpose Layer (Swap Directions)")
        # Styling checkbox slightly to match theme
        self.check_transpose.setStyleSheet("QCheckBox { spacing: 10px; }")
        layout.addWidget(self.check_transpose)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background-color: #44475a;")  # Grey for cancel
        btn_cancel.clicked.connect(self.reject)

        btn_apply = QPushButton("Apply Changes")
        btn_apply.clicked.connect(self.accept)

        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_apply)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def get_selection(self):
        return self.combo_source.currentText(), self.combo_dest.currentText(), self.check_transpose.isChecked()


class MatModifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAT/NPY File Modifier")
        self.resize(900, 700)

        # Data storage
        self.layers_data = None
        self.second_file_data = None

        self.initUI()

    def initUI(self):
        # Main Layout Container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # --- Section 1: File Selection ---
        # We use objectName 'ContentFrame' to target the specific CSS for the card look
        file_frame = QFrame()
        file_frame.setObjectName("ContentFrame")
        file_layout = QVBoxLayout(file_frame)

        # Row 1: Target File
        file_layout.addWidget(QLabel("First File (Target to Modify):"))
        row1 = QHBoxLayout()
        self.entry_target = QLineEdit()
        self.entry_target.setReadOnly(True)
        self.entry_target.setPlaceholderText("No file selected...")
        btn_browse_target = QPushButton("Browse")
        btn_browse_target.clicked.connect(lambda: self.select_file(1))
        row1.addWidget(self.entry_target)
        row1.addWidget(btn_browse_target)
        file_layout.addLayout(row1)

        # Row 2: Source File
        file_layout.addWidget(QLabel("Second File (Source Data):"))
        row2 = QHBoxLayout()
        self.entry_source = QLineEdit()
        self.entry_source.setReadOnly(True)
        self.entry_source.setPlaceholderText("No file selected...")
        btn_browse_source = QPushButton("Browse")
        btn_browse_source.clicked.connect(lambda: self.select_file(2))
        row2.addWidget(self.entry_source)
        row2.addWidget(btn_browse_source)
        file_layout.addLayout(row2)

        main_layout.addWidget(file_frame)

        # --- Section 2: Action ---
        self.btn_process = QPushButton("PROCESS FILES")
        self.btn_process.setMinimumHeight(50)
        self.btn_process.setStyleSheet("font-size: 14pt; letter-spacing: 1px;")
        self.btn_process.clicked.connect(self.process_files)
        main_layout.addWidget(self.btn_process)

        # --- Section 3: Log ---
        log_label = QLabel("System Log:")
        main_layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # Monospace font for logs usually looks better
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
        main_layout.addWidget(self.log_text)

    # ==========================
    # LOGIC METHODS
    # ==========================

    def log_message(self, message, level="INFO"):
        color = "#50fa7b" if level == "INFO" else "#ff5555"  # Green for info, Red for error
        html = f'<span style="color:{color}">[{level}]</span> <span style="color:#f8f8f2">{message}</span>'
        self.log_text.append(html)

    def select_file(self, file_num):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Data Files (*.mat *.npy);;All Files (*)")
        if path:
            data = self.verify_file(path, file_num)
            if data is not None:
                if file_num == 1:
                    self.layers_data = data
                    self.entry_target.setText(path)
                    self.log_message(f"Target file loaded: {os.path.basename(path)}")
                else:
                    self.second_file_data = data
                    self.entry_source.setText(path)
                    self.log_message(f"Source file loaded: {os.path.basename(path)}")

    def load_file_data(self, filepath):
        try:
            file_ext = os.path.splitext(filepath)[1].lower()
            if file_ext == '.mat':
                try:
                    mat = sio.loadmat(filepath)
                except NotImplementedError:
                    # h5py fallback
                    with h5py.File(filepath, 'r') as f:
                        for k in f.keys():
                            if isinstance(f[k], h5py.Dataset): return {k: np.array(f[k])}
                        return {}
                for key in mat:
                    if not key.startswith('__'): return mat
                return None
            elif file_ext == '.npy':
                return {'data': np.load(filepath)}
            return None
        except Exception as e:
            self.log_message(f"Read Error: {e}", "ERROR")
            return None

    def verify_file(self, filepath, file_num):
        data_dict = self.load_file_data(filepath)
        if not data_dict: return None
        for key, val in data_dict.items():
            if not key.startswith('__') and isinstance(val, np.ndarray) and val.ndim >= 2:
                self.log_message(f"Verified File {file_num}. Shape: {val.shape}")
                return val.copy()
        self.log_message(f"No valid data array in File {file_num}", "ERROR")
        return None

    def process_files(self):
        if self.layers_data is None or self.second_file_data is None:
            self.log_message("Please load both files first.", "ERROR")
            return

        # Dimension Check
        s1 = self.layers_data.shape[:2]
        s2 = self.second_file_data.shape[:2]
        if s1 != s2 and s1 != s2[::-1]:
            self.log_message(f"Dimension Mismatch: Target {s1} vs Source {s2}", "ERROR")
            return

        # Open Dialog
        dlg = LayerSelectionDialog(self, self.second_file_data, self.layers_data)
        if dlg.exec_() == QDialog.Accepted:
            src_choice, dest_choice, transpose = dlg.get_selection()
            self.perform_modification(src_choice, dest_choice, transpose)

    def perform_modification(self, source_choice, dest_choice, should_transpose):
        # 1. Extract Source Slice
        s_idx = 0 if "2D" in source_choice else int(source_choice.split(" ")[1]) - 1

        src_slice = self.second_file_data if self.second_file_data.ndim == 2 else self.second_file_data[:, :, s_idx]
        src_slice = src_slice.copy()

        # 2. Transpose if needed
        if should_transpose:
            src_slice = src_slice.T
            self.log_message(f"Transposed source. New shape: {src_slice.shape}")

        # 3. Double check shape match
        if self.layers_data.shape[:2] != src_slice.shape[:2]:
            self.log_message(f"Shape Mismatch! Target {self.layers_data.shape[:2]} vs Source {src_slice.shape[:2]}",
                             "ERROR")
            return

        # 4. Modify / Append
        if dest_choice == "Add as new layer":
            self.layers_data = np.dstack((self.layers_data, src_slice))
            self.log_message(f"Added new layer. Total layers: {self.layers_data.shape[2]}")
        else:
            d_idx = 0 if "2D" in dest_choice else int(dest_choice.split(" ")[1]) - 1
            if self.layers_data.ndim == 2:
                self.layers_data = src_slice
            else:
                self.layers_data[:, :, d_idx] = src_slice
            self.log_message(f"Overwrote destination layer {d_idx + 1}")

        self.save_modified_file()

    def save_modified_file(self):
        path = self.entry_target.text()
        if not path: return

        dir_name = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(dir_name, f"{base_name}_modified.mat")

        try:
            sio.savemat(out_path, {'layers': self.layers_data})
            self.log_message(f"SUCCESS: Saved to {out_path}")
            QMessageBox.information(self, "Success", f"File saved:\n{out_path}")
        except Exception as e:
            self.log_message(f"Save Failed: {e}", "ERROR")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # APPLY THE THEME
    app.setStyleSheet(STYLESHEET)

    window = MatModifierApp()
    window.show()
    sys.exit(app.exec_())