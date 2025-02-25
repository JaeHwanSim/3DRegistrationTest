from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QFileDialog, QHBoxLayout, QLabel,
                           QMessageBox, QApplication)
from PyQt5.QtCore import Qt, QTimer
from viewer import Viewer3D
from model import STLModel
from registration import Registration
import numpy as np

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

        # 버튼 생성 및 레이아웃 추가
        self.load_source_button = QPushButton("소스 모델 불러오기")
        self.load_target_button = QPushButton("타겟 모델 불러오기")
        self.show_viewer_button = QPushButton("모델 보기")
        self.pca_button = QPushButton("PCA 정합")
        self.fpfh_button = QPushButton("FPFH 특징점 추출")
        self.ransac_button = QPushButton("RANSAC 전역 정합")
        self.icp_button = QPushButton("ICP 미세 정합")
        
        # 버튼 비활성화 초기 상태
        # self.load_target_button.setEnabled(False)
        self.show_viewer_button.setEnabled(False)
        self.pca_button.setEnabled(False)
        self.fpfh_button.setEnabled(False)
        self.ransac_button.setEnabled(False)
        self.icp_button.setEnabled(False)
        
        # 버튼 이벤트 연결
        self.load_source_button.clicked.connect(lambda: self.load_stl('source'))
        self.load_target_button.clicked.connect(lambda: self.load_stl('target'))
        self.show_viewer_button.clicked.connect(self.show_viewer)
        self.pca_button.clicked.connect(self.execute_pca_alignment)
        self.fpfh_button.clicked.connect(self.execute_fpfh_extraction)
        self.ransac_button.clicked.connect(self.execute_ransac)
        self.icp_button.clicked.connect(self.execute_icp)
        
        # 버튼 레이아웃에 추가
        button_layout.addWidget(self.load_source_button)
        button_layout.addWidget(self.load_target_button)
        button_layout.addWidget(self.show_viewer_button)
        button_layout.addWidget(self.pca_button)
        button_layout.addWidget(self.fpfh_button)
        button_layout.addWidget(self.ransac_button)
        button_layout.addWidget(self.icp_button)

        layout.addWidget(button_container)

        # 결과 표시 레이블
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        # 초기 버튼 상태 설정
        self.disable_registration_buttons()

    def disable_registration_buttons(self):
        """정합 관련 버튼들 비활성화"""
        for button in [self.show_viewer_button, self.fpfh_button, self.pca_button,
                      self.ransac_button, self.icp_button]:
            button.setEnabled(False)

    def load_stl(self, model_type):
        """STL 파일 불러오기"""
        try:
            # 파일 선택 대화상자
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"{model_type.capitalize()} 모델 선택", "", "STL 파일 (*.stl)")
            
            if not file_path:
                return
                
            # UI 업데이트 중임을 표시
            self.status_label.setText(f"{model_type.capitalize()} 모델 로드 중...")
            QApplication.processEvents()  # UI 업데이트
            
            # 모델 로드
            model = STLModel()
            if model.load(file_path):
                # 모델 객체 저장
                if model_type == 'source':
                    self.source_model = model
                    self.load_source_button.setStyleSheet("background-color: lightgreen")
                    self.status_label.setText("소스 모델 로드 완료")
                    self.source_model.set_color([0, 1, 0.5])
                    # 다음 단계 활성화
                    self.load_target_button.setEnabled(True)
                    self.status_label.setText("2단계: 타겟 모델을 불러오세요")
                else:  # target
                    self.target_model = model
                    self.load_target_button.setStyleSheet("background-color: lightgreen")
                    self.status_label.setText("타겟 모델 로드 완료")
                    self.target_model.set_color([1, 0.5, 0])
                    # 다음 단계 활성화
                    self.show_viewer_button.setEnabled(True)
                    self.status_label.setText("3단계: 모델 보기를 실행하세요")
                
                # 성공 메시지
                self.result_label.setText(f"{model_type.capitalize()} 모델 로드 완료\n정점 수: {len(model.mesh.vertices)}")
                
                # 두 모델이 모두 로드되었는지 확인
                if self.source_model and self.target_model:
                    self.status_label.setText("3단계: 모델 보기를 실행하세요")
            else:
                QMessageBox.critical(self, "오류", f"{model_type.capitalize()} 모델 로드 중 오류가 발생했습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"STL 파일 로드 중 오류가 발생했습니다: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_viewer(self):
        """주기적으로 뷰어 업데이트"""
        if self.viewer and hasattr(self.viewer, 'is_visible') and self.viewer.is_visible:
            self.viewer.update()

    def show_viewer(self):
        """모델 보기"""
        try:
            self.status_label.setText("뷰어 초기화 중...")
            QApplication.processEvents()  # UI 업데이트
            
            # 뷰어가 없으면 새로 생성
            if self.viewer is None:
                self.viewer = Viewer3D()
            
            # 뷰어 초기화
            self.viewer.clear()
            
            # 모델 추가 전 확인
            if self.source_model is None or self.target_model is None:
                QMessageBox.warning(self, "경고", "소스 모델과 타겟 모델을 모두 불러와야 합니다.")
                return
                
            # 타겟 모델 먼저 추가 (레이어 순서)
            self.viewer.add_model(self.target_model)
            QApplication.processEvents()  # UI 업데이트
            
            # 소스 모델 추가
            self.viewer.add_model(self.source_model)
            QApplication.processEvents()  # UI 업데이트
            
            # 뷰어 표시
            self.status_label.setText("뷰어 표시 중...")
            QApplication.processEvents()  # UI 업데이트
            
            self.viewer.show()
            
            # 다음 단계 활성화
            self.pca_button.setEnabled(True)
            self.status_label.setText("4단계: PCA 정합을 실행하세요")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"뷰어 실행 중 오류가 발생했습니다: {str(e)}")
            import traceback
            traceback.print_exc()

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
        """4단계: PCA 정합"""
        try:
            # UI 업데이트
            self.status_label.setText("PCA 정합 실행 중...")
            self.result_label.setText("PCA 계산 중...")
            self.pca_button.setEnabled(False)
            QApplication.processEvents()  # UI 업데이트
            
            # PCA 정합 실행 (시간이 걸리는 작업)
            transformation = self.registration.execute_pca_alignment(
                self.source_model, self.target_model)
            
            # UI 업데이트
            self.status_label.setText("모델 업데이트 중...")
            QApplication.processEvents()  # UI 업데이트
            
            # 결과 메시지 생성
            result_msg = "PCA 정합 완료"
            
            # 결과 표시
            self.result_label.setText(result_msg)
            
            # 모델 시각화 업데이트
            if self.viewer and self.viewer.is_visible:
                self.status_label.setText("뷰어 업데이트 중...")
                QApplication.processEvents()  # UI 업데이트
                
                self.viewer.clear()
                self.viewer.add_model(self.target_model)
                QApplication.processEvents()  # UI 업데이트
                
                self.viewer.add_model(self.source_model)
                QApplication.processEvents()  # UI 업데이트
                
                # 뷰 중심 재설정
                self.viewer.reset_view_to_center()
            
            # 다음 단계 활성화
            self.pca_button.setEnabled(True)
            self.fpfh_button.setEnabled(True)  # FPFH 버튼 활성화
            self.status_label.setText("5단계: FPFH 특징점을 추출하세요")
            
        except Exception as e:
            self.pca_button.setEnabled(True)  # 버튼 재활성화
            QMessageBox.critical(self, "오류", f"PCA 정합 중 오류가 발생했습니다: {str(e)}")
            import traceback
            traceback.print_exc()

    def execute_fpfh_extraction(self):
        """5단계: FPFH 특징점 추출"""
        try:
            # UI 업데이트
            self.status_label.setText("FPFH 특징점 추출 중...")
            self.result_label.setText("포인트 클라우드 다운샘플링 중...")
            self.fpfh_button.setEnabled(False)  # 버튼 비활성화
            QApplication.processEvents()  # UI 업데이트
            
            # FPFH 계산 실행
            result = self.registration.compute_fpfh_features(
                self.source_model, self.target_model)
            
            # UI 업데이트
            self.status_label.setText("특징점 시각화 중...")
            QApplication.processEvents()  # UI 업데이트
            
            # 결과 메시지 생성
            result_msg = (
                f"FPFH 특징점 추출 완료\n"
                f"소스 모델 특징점 수: {result['source_points']}\n"
                f"타겟 모델 특징점 수: {result['target_points']}\n"
                f"소스 중요 특징점 비율: {result['source_key_ratio']:.2%}\n"
                f"타겟 중요 특징점 비율: {result['target_key_ratio']:.2%}"
            )
            
            # 결과 표시
            self.result_label.setText(result_msg)
            
            # 특징점 시각화
            if self.viewer and self.viewer.is_visible:
                self.status_label.setText("뷰어 업데이트 중...")
                QApplication.processEvents()  # UI 업데이트
                
                self.viewer.clear()
                
                # 원래 모델 추가
                self.viewer.add_model(self.target_model)
                QApplication.processEvents()  # UI 업데이트
                
                self.viewer.add_model(self.source_model)
                QApplication.processEvents()  # UI 업데이트
                
                # 특징점 추가
                if hasattr(self.registration, 'source_down'):
                    self.viewer.add_geometry(self.registration.source_down)
                    QApplication.processEvents()  # UI 업데이트
                
                if hasattr(self.registration, 'target_down'):
                    self.viewer.add_geometry(self.registration.target_down)
                    QApplication.processEvents()  # UI 업데이트
                
                # 뷰 중심 재설정
                self.viewer.reset_view_to_center()
            
            # 다음 단계 활성화
            self.fpfh_button.setEnabled(True)  # 버튼 재활성화
            self.ransac_button.setEnabled(True)  # RANSAC 버튼 활성화
            self.status_label.setText("6단계: RANSAC 전역 정합을 실행하세요")
            
        except Exception as e:
            self.fpfh_button.setEnabled(True)  # 버튼 재활성화
            QMessageBox.critical(self, "오류", f"FPFH 특징점 추출 중 오류가 발생했습니다: {str(e)}")
            import traceback
            traceback.print_exc()

    def execute_ransac(self):
        """6단계: RANSAC 전역 정합"""
        try:
            # RANSAC 전역 정합 실행
            result = self.registration.execute_ransac(
                self.source_model, self.target_model)
            
            # 결과 메시지 생성
            result_msg = (
                f"RANSAC 전역 정합 완료\n"
                f"일치하는 점 수: {result['n_points']}\n"
                f"RMSE: {result['rmse']:.4f}\n"
                f"변환 행렬:\n{np.array2string(result['transformation'], precision=4)}"
            )
            
            # 결과 표시
            self.result_label.setText(result_msg)
            
            # 모델 시각화 업데이트
            if self.viewer and self.viewer.is_visible:
                self.viewer.clear()
                self.viewer.add_model(self.target_model)
                self.viewer.add_model(self.source_model)
                
                # 뷰 중심 재설정
                self.viewer.reset_view_to_center()
            
            # 다음 단계 활성화
            self.icp_button.setEnabled(True)  # ICP 버튼 활성화
            self.status_label.setText("7단계: ICP 미세 정합을 실행하세요")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"RANSAC 전역 정합 중 오류가 발생했습니다: {str(e)}")

    def execute_icp(self):
        """7단계: ICP 미세 정합"""
        try:
            # ICP 미세 정합 실행
            result = self.registration.execute_icp(
                self.source_model, self.target_model)
            
            # 결과 메시지 생성
            result_msg = (
                f"ICP 미세 정합 완료\n"
                f"정합 점 수: {result['n_points']}\n"
                f"RMSE: {result['rmse']:.4f}\n"
                f"반복 횟수: {result['iteration']}\n"
                f"변환 행렬:\n{np.array2string(result['transformation'], precision=4)}"
            )
            
            # 결과 표시
            self.result_label.setText(result_msg)
            
            # 모델 시각화 업데이트
            if self.viewer and self.viewer.is_visible:
                self.viewer.clear()
                self.viewer.add_model(self.target_model)
                self.viewer.add_model(self.source_model)
                
                # 뷰 중심 재설정
                self.viewer.reset_view_to_center()
            
            self.status_label.setText("정합 완료! 모든 단계가 완료되었습니다.")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"ICP 미세 정합 중 오류가 발생했습니다: {str(e)}")

    # 여기에 나머지 단계별 실행 함수들이 추가되어야 합니다