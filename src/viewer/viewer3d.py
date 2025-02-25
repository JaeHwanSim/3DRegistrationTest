import open3d as o3d
import threading
import time
import numpy as np
import traceback

class Viewer3D:
    def __init__(self):
        # 로그 레벨 조정 (경고 메시지 출력 안함)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Viewer", width=800, height=600)
        
        # 렌더 옵션 설정
        opt = self.vis.get_render_option()
        opt.background_color = [1, 1, 1]  # 흰색 배경
        opt.show_coordinate_frame = True  # 좌표축 표시
        opt.point_size = 1.0
        
        self.geometries = []
        self.is_visible = False
        self.running = False
        self.render_thread = None
        
        # 초기 카메라 설정
        self.setup_camera()
        
    def setup_camera(self):
        """초기 카메라 설정"""
        try:
            ctr = self.vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0, 0, -1])   # 앞쪽 방향
            ctr.set_lookat([0, 0, 0])   # 중앙 바라보기
            ctr.set_up([0, -1, 0])      # 위쪽 방향
        except Exception as e:
            print(f"카메라 설정 오류: {e}")
        
    def reset_view_to_center(self):
        """모델들의 중심을 기준으로 카메라 재설정"""
        try:
            if not self.geometries:
                return
                
            # 간단한 방법으로 바운딩 박스 재설정
            self.vis.reset_view_point(True)
            
            # Open3D의 자동 기능 활용
            params = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            self.vis.get_view_control().convert_from_pinhole_camera_parameters(params)
            
            self.update()
        except Exception as e:
            print(f"뷰 리셋 오류: {e}")
            traceback.print_exc()
        
    def add_model(self, model):
        """모델 추가"""
        try:
            if model is None or model.mesh is None:
                print("유효하지 않은 모델")
                return
                
            # 모델 추가 전에 완전히 업데이트
            if self.is_visible:
                self.vis.poll_events()
                self.vis.update_renderer()
                
            # 지오메트리 추가
            success = self.vis.add_geometry(model.mesh, reset_bounding_box=False)
            if not success:
                print("지오메트리 추가 실패")
                return
                
            self.geometries.append(model.mesh)
            
            # 업데이트
            if self.is_visible:
                self.vis.update_geometry(model.mesh)
                self.vis.poll_events()
                self.vis.update_renderer()
                
            # 지연 후 뷰 리셋 (이미 실행 중인 경우에만)
            if self.is_visible and len(self.geometries) >= 2:
                # 간단하게 뷰포인트만 재설정
                self.vis.reset_view_point(True)
                self.vis.poll_events()
                self.vis.update_renderer()
                
            print(f"모델 추가됨 (점: {len(model.mesh.vertices)})")
            
        except Exception as e:
            print(f"모델 추가 오류: {e}")
            traceback.print_exc()
            
    def add_geometry(self, geometry):
        """포인트 클라우드나 메쉬 추가"""
        try:
            if geometry:
                self.vis.add_geometry(geometry, reset_bounding_box=False)
                self.geometries.append(geometry)
                # 업데이트만 수행
                self.update()
        except Exception as e:
            print(f"지오메트리 추가 오류: {e}")
            traceback.print_exc()
    
    def clear(self):
        """모든 지오메트리 제거"""
        try:
            for geometry in self.geometries:
                self.vis.remove_geometry(geometry, reset_bounding_box=False)
            self.geometries.clear()
            # 카메라 초기화
            self.setup_camera()
            self.update()
        except Exception as e:
            print(f"뷰어 초기화 오류: {e}")
            traceback.print_exc()
    
    def update(self):
        """뷰어 업데이트"""
        try:
            if self.is_visible:
                for geometry in self.geometries:
                    self.vis.update_geometry(geometry)
                self.vis.poll_events()
                self.vis.update_renderer()
        except Exception as e:
            print(f"업데이트 오류: {e}")
            traceback.print_exc()
    
    def show(self):
        """뷰어 실행 (비블로킹 방식)"""
        try:
            if self.running:
                # 이미 실행 중이면 새로고침만 수행
                self.update()
                return
                
            # 뷰어 표시 및 초기화
            self.is_visible = True
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # 비동기 렌더링 시작
            self.running = True
            self.render_thread = threading.Thread(target=self._render_loop)
            self.render_thread.daemon = True  # 메인 프로그램 종료 시 자동 종료
            self.render_thread.start()
            
            # 뷰 초기화 (모델이 있는 경우)
            if self.geometries:
                # 안전한 방법으로 뷰 재설정
                self.vis.reset_view_point(True)
        except Exception as e:
            print(f"뷰어 실행 오류: {e}")
            traceback.print_exc()
    
    def _render_loop(self):
        """렌더링 루프 (별도 스레드에서 실행)"""
        try:
            while self.running:
                if not self.vis.poll_events():
                    self.running = False
                    self.is_visible = False
                    break
                self.vis.update_renderer()
                # 프레임 제한
                time.sleep(0.01)
        except Exception as e:
            print(f"렌더링 루프 오류: {e}")
            traceback.print_exc()
            self.running = False
            self.is_visible = False
    
    def close(self):
        """뷰어 종료"""
        try:
            self.running = False
            if self.render_thread and self.render_thread.is_alive():
                self.render_thread.join(timeout=1.0)
            self.vis.destroy_window()
            self.is_visible = False
        except Exception as e:
            print(f"뷰어 종료 오류: {e}")
            traceback.print_exc()
        