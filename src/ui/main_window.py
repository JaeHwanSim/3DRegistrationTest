from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QFileDialog, QHBoxLayout, QLabel,
                           QMessageBox)
from PyQt5.QtCore import Qt
from viewer import Viewer3D
from model import STLModel
from registration import Registration

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = None
        self.registration = Registration()
        self.source_model = None
        self.target_model = None
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("치아 모델 정합 시스템")
        self.setGeometry(100, 100, 800, 600)

        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 상태 표시 레이블
        self.status_label = QLabel("파일을 로드해주세요")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # 버튼 컨테이너
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)

        # 소스/타겟 모델 로드 버튼 그룹
        model_buttons = QWidget()
        model_layout = QHBoxLayout(model_buttons)

        # 소스 모델 로드 버튼
        self.source_button = QPushButton("소스 모델 로드")
        self.source_button.setFixedSize(200, 40)
        self.source_button.clicked.connect(lambda: self.load_stl("source"))
        model_layout.addWidget(self.source_button)

        # 타겟 모델 로드 버튼
        self.target_button = QPushButton("타겟 모델 로드")
        self.target_button.setFixedSize(200, 40)
        self.target_button.clicked.connect(lambda: self.load_stl("target"))
        model_layout.addWidget(self.target_button)

        button_layout.addWidget(model_buttons)

        # 뷰어 표시 버튼
        self.view_button = QPushButton("모델 보기")
        self.view_button.setFixedSize(200, 40)
        self.view_button.clicked.connect(self.show_viewer)
        self.view_button.setEnabled(False)
        button_layout.addWidget(self.view_button, alignment=Qt.AlignCenter)

        # 정합 버튼 추가
        self.register_button = QPushButton("정합 실행")
        self.register_button.setFixedSize(200, 40)
        self.register_button.clicked.connect(self.perform_registration)
        self.register_button.setEnabled(False)
        button_layout.addWidget(self.register_button, alignment=Qt.AlignCenter)

        # 결과 표시 레이블 추가
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        layout.addWidget(button_container)

    def load_stl(self, model_type):
        file_name, _ = QFileDialog.getOpenFileName(
            self, f"{model_type} STL 파일 선택", "", "STL Files (*.stl)")
        
        if file_name:
            if model_type == "source":
                self.source_model = STLModel()
                if self.source_model.load(file_name):
                    self.source_model.mesh.paint_uniform_color([1, 0, 0])  # 빨간색
                    self.source_button.setStyleSheet("background-color: lightgreen")
                    self.status_label.setText("소스 모델 로드 완료")
            else:
                self.target_model = STLModel()
                if self.target_model.load(file_name):
                    self.target_model.mesh.paint_uniform_color([0, 0, 1])  # 파란색
                    self.target_button.setStyleSheet("background-color: lightgreen")
                    self.status_label.setText("타겟 모델 로드 완료")

            # 두 모델이 모두 로드되었는지 확인
            if self.source_model and self.target_model:
                self.view_button.setEnabled(True)
                self.register_button.setEnabled(True)
                self.status_label.setText("두 모델 모두 로드 완료. 정합을 시작할 수 있습니다.")

    def show_viewer(self):
        if self.viewer is None:
            self.viewer = Viewer3D()
        
        if self.source_model:
            self.viewer.add_model(self.source_model)
        if self.target_model:
            self.viewer.add_model(self.target_model)
        
        self.viewer.show()
    
    def closeEvent(self, event):
        """윈도우 종료 시 처리"""
        if self.viewer:
            self.viewer.close()
        event.accept()

    def perform_registration(self):
        try:
            # 정합 실행
            result = self.registration.register(self.source_model, self.target_model)
            
            # 결과 메시지 생성
            result_msg = (f"정합 완료\n"
                         f"정확도: {result['fitness']:.4f}\n"
                         f"RMSE: {result['inlier_rmse']:.4f}")
            
            # 결과 표시
            self.result_label.setText(result_msg)
            
            # 뷰어 업데이트
            if self.viewer:
                self.viewer.clear()  # 기존 모델 제거
                self.viewer.add_model(self.target_model)  # 타겟 모델 추가
                self.viewer.add_model(self.source_model)  # 변환된 소스 모델 추가
                self.viewer.update()
            
            # 성공 메시지 표시
            QMessageBox.information(self, "정합 완료", "모델 정합이 완료되었습니다.")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"정합 중 오류가 발생했습니다: {str(e)}")