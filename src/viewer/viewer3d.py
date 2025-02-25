import open3d as o3d
import threading
import time

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
        
        self.setup_camera()
        self.geometries = []
        self.is_visible = False
        self.running = False
        self.render_thread = None
        self.render_lock = threading.Lock()  # 스레드 동기화를 위한 락 추가
        
    def setup_camera(self):
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
    def add_model(self, model):
        if model and model.mesh:
            self.vis.add_geometry(model.mesh, reset_bounding_box=True)
            self.geometries.append(model.mesh)
            self.update()
            
    def add_geometry(self, geometry):
        """포인트 클라우드나 메쉬 추가"""
        if geometry:
            self.vis.add_geometry(geometry, reset_bounding_box=False)
            self.geometries.append(geometry)
            self.update()
    
    def clear(self):
        """모든 지오메트리 제거"""
        for geometry in self.geometries:
            self.vis.remove_geometry(geometry, reset_bounding_box=False)
        self.geometries.clear()
    
    def update(self):
        """뷰어 업데이트"""
        if self.is_visible:
            for geometry in self.geometries:
                self.vis.update_geometry(geometry)
            self.vis.poll_events()
            self.vis.update_renderer()
    
    def show(self):
        """뷰어 실행 (비블로킹 방식)"""
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
    
    def _render_loop(self):
        """렌더링 루프 (별도 스레드에서 실행)"""
        while self.running:
            if not self.vis.poll_events():
                self.running = False
                self.is_visible = False
                break
            self.vis.update_renderer()
            # 프레임 제한
            time.sleep(0.01)
    
    def close(self):
        """뷰어 종료"""
        self.running = False
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
        self.vis.destroy_window()
        self.is_visible = False
        