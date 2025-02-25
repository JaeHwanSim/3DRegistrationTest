from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QFileDialog, QHBoxLayout, QLabel,
                           QMessageBox)
from PyQt5.QtCore import Qt, QTimer
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
        
        # 뷰어 업데이트를 위한 타이머
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_viewer)
        self.update_timer.start(100)  # 100ms 간격으로 업데이트
        
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("치아 모델 정합 시스템")
        self.setGeometry(100, 100, 800, 600)

        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 상태 표시 레이블
        self.status_label = QLabel("1단계: STL 파일을 로드해주세요")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # 버튼들을 담을 컨테이너
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)

        # 1-2단계: 파일 로드 버튼들
        file_widget = QWidget()
        file_layout = QHBoxLayout(file_widget)
        self.source_button = QPushButton("1. 소스 STL 로드")
        self.target_button = QPushButton("2. 타겟 STL 로드")
        self.source_button.clicked.connect(lambda: self.load_stl("source"))
        self.target_button.clicked.connect(lambda: self.load_stl("target"))
        file_layout.addWidget(self.source_button)
        file_layout.addWidget(self.target_button)
        button_layout.addWidget(file_widget)

        # 3-8단계: 정합 과정 버튼들
        self.view_button = QPushButton("3. 모델 보기")
        self.pca_button = QPushButton("4. PCA 기반 주축 정렬")
        self.fpfh_button = QPushButton("5. FPFH 특징점 추출")
        self.ransac_button = QPushButton("6. RANSAC 전역 정합")
        self.rough_icp_button = QPushButton("7. ICP 거친 정합")
        self.fine_icp_button = QPushButton("8. ICP 정밀 정합")

        # 버튼 이벤트 연결
        self.view_button.clicked.connect(self.show_viewer)
        self.pca_button.clicked.connect(self.execute_pca_alignment)
        # self.fpfh_button.clicked.connect(self.execute_fpfh_extraction)
        # self.ransac_button.clicked.connect(self.execute_ransac)
        # self.rough_icp_button.clicked.connect(self.execute_rough_icp)
        # self.fine_icp_button.clicked.connect(self.execute_fine_icp)
        
        self.pca_button.setEnabled(True)

        # 버튼들 추가
        for button in [self.view_button, self.pca_button, self.fpfh_button,
                      self.ransac_button, self.rough_icp_button, self.fine_icp_button]:
            button.setFixedHeight(40)
            button_layout.addWidget(button)

        layout.addWidget(button_container)

        # 결과 표시 레이블
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        # 초기 버튼 상태 설정
        self.disable_registration_buttons()

    def disable_registration_buttons(self):
        """정합 관련 버튼들 비활성화"""
        for button in [self.view_button, self.pca_button, self.fpfh_button,
                      self.ransac_button, self.rough_icp_button, self.fine_icp_button]:
            button.setEnabled(False)

    def load_stl(self, model_type):
        """STL 파일 로드"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, f"{model_type} STL 파일 선택", "", "STL Files (*.stl)")
        
        if file_name:
            if model_type == "source":
                self.source_model = STLModel()
                if self.source_model.load(file_name):
                    self.source_button.setStyleSheet("background-color: lightgreen")
                    self.status_label.setText("소스 모델 로드 완료")
                    self.source_model.set_color([0, 1, 0.5])
            else:
                self.target_model = STLModel()
                if self.target_model.load(file_name):
                    self.target_button.setStyleSheet("background-color: lightgreen")
                    self.status_label.setText("타겟 모델 로드 완료")
                    self.target_model.set_color([1, 0.5, 0])

            # 두 모델이 모두 로드되었는지 확인
            if self.source_model and self.target_model:
                self.view_button.setEnabled(True)
                self.status_label.setText("3단계: 모델 보기를 실행하세요")

    def update_viewer(self):
        """주기적으로 뷰어 업데이트"""
        if self.viewer and hasattr(self.viewer, 'is_visible') and self.viewer.is_visible:
            self.viewer.update()

    def show_viewer(self):
        """모델 보기"""
        try:
            if self.viewer is None:
                self.viewer = Viewer3D()
            if self.source_model and self.target_model:
                self.viewer.clear()
                self.source_model.mesh.paint_uniform_color([0, 1, 0.5])
                self.target_model.mesh.paint_uniform_color([1, 0.5, 0])
                self.viewer.add_model(self.target_model)
                self.viewer.add_model(self.source_model)
                self.viewer.show()
                self.pca_button.setEnabled(True)
                self.status_label.setText("4단계: PCA 기반 주축 정렬을 실행하세요")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"뷰어 실행 중 오류가 발생했습니다: {str(e)}")

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

    def execute_pca_alignment(self):
        """4단계: PCA 기반 주축 정렬"""
        try:
            # PCA 정렬 실행
            self.registration.execute_pca_alignment(self.source_model, self.target_model)
            
            # 뷰어 업데이트
            if self.viewer and self.viewer.is_visible:
                self.viewer.clear()
                self.viewer.add_model(self.target_model)
                self.viewer.add_model(self.source_model)
            
            # 다음 단계 활성화
            self.fpfh_button.setEnabled(True)
            self.status_label.setText("5단계: FPFH 특징점 추출을 실행하세요")
            
            # 결과 메시지
            self.result_label.setText("PCA 기반 주축 정렬 완료")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"PCA 정렬 중 오류가 발생했습니다: {str(e)}")

    # 여기에 나머지 단계별 실행 함수들이 추가되어야 합니다